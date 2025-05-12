#!/usr/bin/env python3
import os
import json
import base64
import boto3
import numpy as np
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import SubscribeToTopicRequest
from facenet_pytorch import MTCNN
from PIL import Image
import time

# Configuration Constants
TOPIC_NAME = os.environ["TOPIC_NAME"]
REQUEST_QUEUE_URL = os.environ["REQUEST_QUEUE_URL"]
RESPONSE_QUEUE_URL = os.environ["RESPONSE_QUEUE_URL"]
TMP_DIR = os.environ["TMP_DIR"]

# AWS Clients Initialization
ipc_client = GreengrassCoreIPCClientV2()
sqs = boto3.client("sqs")
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
seen_req_ids = set()

# Face Detection Class
class face_detection:
    """
    Face Detection Class using MTCNN Model
    """
    def __init__(self):
        """
        Initializes the face detection object with MTCNN model.
        """
        self.mtcnn = mtcnn
    def face_detection_func(self, test_image_path, output_path):
        """
        Detects the face in the given image path and saves it to an output directory.
        Parameters:
            test_image_path : str
                Path to the input image.
            output_path : str
                Directory where detected face image will be saved. Directory is created if it does not exist.
        Returns:
            str: Path to the saved face image, or None if no face is detected.
        """
        # Load image and convert to PIL Image
        img = Image.open(test_image_path).convert("RGB")
        img = np.array(img)
        img = Image.fromarray(img)
        
        # Get the key from the filename
        key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]
        
        # Run MTCNN for face detection.
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)
        
        # If face is detected, save it to the output directory.
        # If no face is detected, return None.
        if face is not None:
            # Create output directory if it does not exist.
            os.makedirs(output_path, exist_ok=True)
            
            # Normalize and prepare the detected face image
            face_img = face - face.min()
            face_img = face_img / face_img.max()
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
            face_pil = Image.fromarray(face_img, mode="RGB")
            
            # Save the detected face image
            face_img_path = os.path.join(output_path, f"{key}_face.jpg")
            face_pil.save(face_img_path)
            
            # Return the path to the saved face image
            return face_img_path
        else:
            return None

# Instantiate face detection object
fd = face_detection()


def process_message(message_payload: str):
    """
    Processes the message payload received from the Greengrass Topic.
    Parameters:
        message_payload: str
            Raw JSON string received from the IPC subscription.
    """
    body = json.loads(message_payload)
    
    # Extract the encoded image and request ID from the message payload.
    encoded = body["encoded"]
    print(f"[FD] Received message for {encoded}")
    req_id = body["request_id"]
    fname = body.get("filename", "")

    # If the request ID has already been processed, return.
    if req_id in seen_req_ids:
        return
    seen_req_ids.add(req_id)

    # 1) Decode JPEG and save to temporary file.
    raw_bytes = base64.b64decode(encoded)
    in_path = os.path.join(TMP_DIR, f"{req_id}.jpg")
    with open(in_path, "wb") as f:
        f.write(raw_bytes)
    
    # 2) Run face detection
    temp_output_path = "/tmp/detected_faces"
    os.makedirs(temp_output_path, exist_ok=True)
    
    face_path = fd.face_detection_func(in_path, temp_output_path)

    # Bonus Section
    # If no face is detected, send the response directly to the response queue.
    if face_path is None:
        print(f"[FD] No face detected for {req_id}")

        response_payload = {
            "request_id": req_id,
            "result": "No-Face",
            "filename": fname,
        }
        sqs.send_message(QueueUrl=RESPONSE_QUEUE_URL, MessageBody=json.dumps(response_payload))
        
        print(f"[FD] Sent 'No-Face' response for {req_id} to response queue")
        return

    # If face is detected, encode the face image.
    with open(face_path, 'rb') as f:
        face_data = f.read()
    face_base64 = base64.b64encode(face_data).decode('utf-8')

    # 3) Send the face image to the face recognition component.
    payload = {
        "request_id": req_id,
        "face_image": face_base64,
        "filename": fname,
    }
    sqs.send_message(QueueUrl=REQUEST_QUEUE_URL, MessageBody=json.dumps(payload))
    print(f"[FD] Sent SQS message for {req_id}")


def on_stream_event(evt):
    """
    Handles the event received from the Greengrass IPC subscription.
    Parameters:
        evt: awsiot.greengrasscoreipc.model.SubscribeToTopicEvent
            Event wrapper containing the binary message that was published on the
            topic.
    """
    try:
        print(f"[FD] Received event: {evt}")
        raw = evt.binary_message.message
        process_message(raw.decode("utf-8"))
    except Exception as e:
        print(f"[FD][Error] {e}")


def on_stream_error(err):
    """Log errors raised by the IPC subscription stream."""
    print(f"[FD][SubscribeError] {err}")

def on_stream_closed() -> None:
    """Log when the IPC subscription stream is closed by Greengrass Core."""
    print('Subscribe to topic stream closed.')

def main():
    """Entry point — establish IPC subscription and keep the script alive."""

    # Subscribe to the configured Greengrass topic.
    _, operation = ipc_client.subscribe_to_topic(topic=TOPIC_NAME, on_stream_event=on_stream_event, on_stream_error=on_stream_error, on_stream_closed=on_stream_closed)
    print('Successfully subscribed to topic: ' + TOPIC_NAME)
    
    # Wait for the subscription to start.
    print("[FD] Subscription started, waiting for messages…")
    
    # Keep the script running.
    try:
        while True:
            time.sleep(10)
    except InterruptedError:
        print('Subscribe interrupted.')


# Main function to run the face detection component
if __name__ == "__main__":
    main()
