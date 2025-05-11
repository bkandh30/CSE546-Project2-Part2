# CSE 546 Cloud Computing - Project 2 (Part II): Edge Computing

## Project Summary

This project focuses on migrating a segment of a face recognition pipeline from a cloud-centric AWS Lambda implementation (developed in Part I) to an edge computing architecture. The core objective is to leverage AWS IoT and edge computing services to perform computations closer to the data source (emulated IoT devices). This approach aims to enable real-time responses, reduce data transfer to the remote cloud, and enhance privacy by processing raw data on user-trusted devices. While real IoT hardware is not used, EC2 instances emulate these devices, interacting with genuine AWS IoT services like IoT Greengrass and MQTT.

## Architecture

The application implements a distributed pipeline to recognize faces in video frames collected from emulated IoT devices (e.g., smart cameras). The data flows through the system as follows:

1.  **Video Frame Ingestion:** An IoT device (emulated by an EC2 instance named "IoT-Greengrass-Client") captures/generates video frames and publishes them via MQTT to a specific topic.
2.  **Edge-based Face Detection:** An AWS IoT Greengrass Core device (emulated by an EC2 instance named "IoT-Greengrass-Core") subscribes to the MQTT topic. A custom Greengrass component running on this core device:
    - Receives the video frames.
    - Performs face detection using a Machine Learning model (MTCNN - Multi-task Cascaded Convolutional Networks).
    - Produces the detected face data.
3.  **Request to Cloud:** The detected faces (or a "No-Face" indicator, see Bonus Feature) are sent as messages to an Amazon SQS (Simple Queue Service) request queue.
4.  **Cloud-based Face Recognition:** The SQS message triggers an AWS Lambda function. This function:
    - Performs face recognition on the detected faces using an ML model (FaceNet).
    - Produces classification results (names of recognized individuals).
5.  **Result Retrieval:** The recognition results are sent to an SQS response queue. The original IoT client device retrieves these results from the response queue.

## Technologies Used

- **AWS IoT Greengrass (v2):** Enables local compute, messaging, data caching, sync, and ML inference capabilities on edge devices.
- **AWS Lambda:** Serverless compute service for running the face recognition model in the cloud.
- **Amazon SQS (Simple Queue Service):** Managed message queuing service for decoupling the edge and cloud components.
- **MQTT (Message Queuing Telemetry Transport):** Lightweight messaging protocol for communication between the IoT client device and the Greengrass Core.
- **Amazon EC2:** Used to emulate the IoT Greengrass Core device and the IoT client device.
- **Machine Learning Models:**
  - **MTCNN (Multi-task Cascaded Convolutional Networks):** For face detection on the edge.
  - **FaceNet:** For face recognition in the cloud.
- **Python:** Primary programming language for the Greengrass component and Lambda function.
- **AWS CLI:** Command-line interface for interacting with AWS services.
- **AWS IoT Device SDK:** Used by the client device to interact with AWS IoT services.

## Setup & Configuration

### 1. Setting up the IoT Greengrass Core Device (`IoT-Greengrass-Core`)

- Launched a `t2.micro` EC2 instance with Amazon Linux 2023.
- Installed Java runtime (OpenJDK 11).
- Created a dedicated system user (`ggc_user`) and group (`ggc_group`) for running Greengrass components.
- Configured AWS credentials and region (`us-east-1`).
- Downloaded, installed, and configured the AWS IoT Greengrass Core software. This involved provisioning AWS resources (IoT thing, thing group, policies, IAM roles, and role aliases) required for the core device to operate.

### 2. Creating the Face Detection Greengrass Component (`com.clientdevices.FaceDetection`)

- Developed a custom Greengrass component with a recipe and artifacts.
- **Recipe (`com.clientdevices.FaceDetection-1.0.0.json`):**
  - Defined component metadata, parameters, dependencies, and lifecycle hooks.
  - Specified installation of Python dependencies (`awsiotsdk`, `boto3`, `numpy`, `torch`, `torchvision`, `torchaudio`, and `pillow`) using `pip`.
  - Configured the component to run a Python script (`fd_component.py`).
- **Artifacts (`fd_component.py` and `facenet_pytorch` module):**
  - The Python script subscribes to an MQTT topic (`clients/<ASU-ID>-IoTThing`).
  - It extracts Base64-encoded image data, `request_id`, and `filename` from incoming JSON messages.
  - Performs face detection using the MTCNN model (via the provided `facenet_pytorch` code).
  - If faces are detected, it sends the processed face data to the SQS request queue (`<ASU-ID>-req-queue`).
- Deployed the component to the Greengrass Core device.

### 3. Setting up the IoT Greengrass Client Device (`IoT-Greengrass-Client`)

- Launched a `t2.micro` EC2 instance with Ubuntu.
- Installed and configured the AWS CLI with appropriate credentials.
- Created AWS IoT resources for the client device:
  - An IoT thing object named `<ASU-ID>-IoTThing`.
  - Generated and managed X.509 certificates and keys for device authentication, attaching them to the thing.
  - Ensured an appropriate IoT policy (e.g., `GreengrassV2IoTThingPolicy`) was attached, granting necessary permissions (Connect, Publish, Subscribe, Receive for IoT Core; `greengrass:*` for Greengrass operations).
- Associated the client device's IoT thing with the Greengrass Core device through the AWS IoT console to enable cloud discovery.
- Configured and deployed necessary client device support components on the Greengrass Core:
  - `aws.greengrass.Nucleus`
  - `aws.greengrass.clientdevices.Auth` (configured with the client device's thing name)
  - `aws.greengrass.clientdevices.mqtt.Moquette` (local MQTT broker)
  - `aws.greengrass.clientdevices.mqtt.Bridge` (configured to relay MQTT messages on the `clients/<ASU-ID>-IoTThing` topic filter from local MQTT to AWS IoT Core and PubSub).
  - `aws.greengrass.clientdevices.IPDetector`
- Installed the AWS IoT Device SDK v2 for Python on the client EC2 instance.

### 4. Simulating Video Frame Generation

- The client device (via the autograder's workload generator) publishes JSON-formatted messages to the MQTT topic `clients/<ASU-ID>-IoTThing`.
- Each message contains a Base64-encoded image (`encoded`), a unique `request_id`, and the `filename`.

## Bonus Feature Implemented

The project successfully implemented the bonus requirement:

- If the face detection component on the Greengrass Core device processes a video frame and detects **no faces**, it directly pushes a response to the SQS response queue (`<ASU-ID>-resp-queue`).
- This response payload contains a 'result' key with the value "No-Face".
- This optimization improves response time and conserves cloud resources (Lambda invocations, SQS messages for recognition) when no faces are present.
