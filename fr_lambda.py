import os
import json
import boto3
import base64
import torch
import numpy as np
from PIL import Image
import asyncio
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sqs = boto3.client('sqs')

class face_recognition:
    async def face_recognition_func(self, model_path, model_wt_path, face_img_path):
        logger.info(f"Loading face image from {face_img_path}")
        face_pil = Image.open(face_img_path).convert("RGB")
        
        key = os.path.splitext(os.path.basename(face_img_path))[0].split(".")[0]
        logger.info(f"Processing image with key: {key}")
        
        logger.info("Converting image to numpy array and normalizing")
        face_numpy = np.array(face_pil, dtype=np.float32) / 255.0
        face_numpy = np.transpose(face_numpy, (2, 0, 1))
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)

        logger.info(f"Loading model weights from {model_wt_path}")
        saved_data = torch.load(model_wt_path)
        logger.info(f"Loading TorchScript model from {model_path}")
        self.resnet = torch.jit.load(model_path)

        if face_tensor is not None:
            logger.info("Computing face embedding")
            emb = self.resnet(face_tensor.unsqueeze(0)).detach()
            embedding_list = saved_data[0]
            name_list = saved_data[1]
            
            logger.info("Calculating distances between input embedding and database embeddings")
            dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
            min_index = dist_list.index(min(dist_list))
            recognized_name = name_list[min_index]
            logger.info(f"Recognized face as: {recognized_name} with distance: {min(dist_list)}")
            return recognized_name
        else:
            logger.info("No face is detected in the provided image")
            return None


fr = face_recognition()
logger.info("Invoking face recognition function")

async def async_handler(event):
    logger.info("Face recognition Lambda invoked")
    model_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), 'resnetV1.pt')
    model_wt_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), 'resnetV1_video_weights.pt')
    logger.info(f"Model path: {model_path}, Model weights path: {model_wt_path}")

    for record in event['Records']:
        logger.info(f"Processing record: {record}")
        message_body = json.loads(record['body'])
        request_id = message_body['request_id']
        face_base64 = message_body['face_image']
        filename = message_body['filename']
        logger.info(f"Record request_id: {request_id}, filename: {filename}")

        temp_face_path = '/tmp/face.jpg'
        logger.info(f"Writing face image to temporary path: {temp_face_path}")
        with open(temp_face_path, 'wb') as f:
            f.write(base64.b64decode(face_base64))
        logger.info("Temporary face image written successfully.")

        recognized_name = await fr.face_recognition_func(model_path, model_wt_path, temp_face_path)
        logger.info(f"Face recognition result for request_id {request_id}: {recognized_name}")

        result_message = {
            'request_id': request_id,
            'result': recognized_name if recognized_name else "Unknown"
        }
        logger.info(f"Sending result to SQS: {result_message}")
        response = sqs.send_message(
            QueueUrl=os.environ["RESPONSE_QUEUE_URL"],
            MessageBody=json.dumps(result_message)
        )
        logger.info(f"SQS response: {response}")

    logger.info("All records processed, returning success response.")
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Face recognition completed successfully'})
    }

def handler(event, context):
    logger.info("Lambda handler invoked for face recognition")
    try:
        result = asyncio.run(async_handler(event))
        logger.info("Async handler executed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Error in face recognition: {str(e)}'})
        }
