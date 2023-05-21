import sys
import os
import cv2
import numpy as np
import socket
import struct
import json
from PIL import Image
from super_gradients.training import models
from super_gradients.common.object_names import Models

# Function to receive images from a socket stream
def receive_image(socket):
    # Read the length of the image as a 4-byte integer
    image_len_data = socket.recv(4)
    image_len = int.from_bytes(image_len_data, byteorder='little', signed=False)
    print("Receiving image of size:", image_len, "bytes")
    # Read the image data
    image_data = b''
    while len(image_data) < image_len:
        received_data = socket.recv(image_len - len(image_data))
        image_data += received_data
        print("Received", len(received_data), "bytes")
    # Convert the image data to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)
    print(f"Len of nparr {len(nparr)}")
    # Decode the image from JPEG format using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print("Failed to decode image")
        return None
    # Convert the image to PIL Image format
    image = Image.fromarray(image)
    return image

# Create a socket object
socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Define the host and port to bind the socket
host = 'localhost'  # Replace with the appropriate host
port = 8080  # Replace with the appropriate port
# Bind the socket to the host and port
socket_server.bind((host, port))
# Set the socket to listen for incoming connections
socket_server.listen(1)

# Create the directory for saving images if it doesn't exist
os.makedirs("cv-images", exist_ok=True)
datadir = "cv-images"

# Initialize the YOLO-NAS model
net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

modelExecute = 0
maxExecutions = 100

# Accept a connection from a client
print("Waiting for a connection...")
socket_client, client_address = socket_server.accept()
print("Connected by", client_address)

# Initialize the frame number
frame_number = 0

# Main loop to receive and process images
while True:
    try:
        # Receive an image from the client
        image = receive_image(socket_client)
        print("Image received!")
        if modelExecute < maxExecutions:
            # Perform inference on the image
            predictions = net.predict(image)
            predictions._images_prediction_lst = list(predictions._images_prediction_lst)
            print("Inference completed!")
            predictions.save(output_folder=datadir + "/" + str(frame_number))
            # predictions.show()
            print(predictions[0])
            identified_objects = []
            for image_prediction in predictions:
                class_names = image_prediction.class_names
                labels = image_prediction.prediction.labels
                confidence = image_prediction.prediction.confidence
                bboxes = image_prediction.prediction.bboxes_xyxy

                for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
                    print("prediction: ", i)
                    print("label_id: ", label)
                    print("label_name: ", class_names[int(label)])
                    print("confidence: ", conf)
                    print("bbox: ", bbox)
                    # identified_objects.append({
                    #     "label_id": int(label),
                    #     "label_name": class_names[int(label)],
                    #     "confidence": int(conf * 100),
                    #     "bbox": bbox
                    # })
                    identified_objects.append({
                        "label_id": int(label),
                        "label_name": class_names[int(label)],
                        "confidence": int(conf * 100)
                    })
                    print("--" * 10)

            # Extract relevant details from the prediction
            # class_names = predictions.class_names
            # labels = predictions.prediction.labels
            # confidence = predictions.prediction.confidence
            # bboxes = predictions.prediction.bboxes_xyxy

            # # Print the extracted details
            # print("Class Names:", class_names)
            # print("Labels:", labels)
            # print("Confidence:", confidence)
            # print("Bounding Boxes:", bboxes)

            # Save label names for the identified objects
            # identified_objects = []
            # for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
            #     label_name = class_names[int(label)]
            #     identified_objects.append(label_name)
            # Convert the identified objects to JSON
            identified_objects_json = json.dumps(identified_objects)
            # Get the size of the JSON data in bytes
            data_size = len(identified_objects_json)
            # Convert the data size to a 4-byte integer
            size_bytes = data_size.to_bytes(4, byteorder='little', signed=False)
            # Send the data size to the client
            socket_client.sendall(size_bytes)
            # Send the identified objects data to the client
            socket_client.sendall(identified_objects_json.encode())
            print(f"Transmitted json data {data_size} bytes")
            modelExecute += 1
            frame_number += 1
            # Convert the identified objects to JSON
            # identified_objects_json = json.dumps(identified_objects)
            # Send the identified objects back to the client
            # socket_client.sendall(identified_objects_json.encode())

    except KeyboardInterrupt:
        # Break the loop if Ctrl+C is pressed
        break

# Close the socket connections
socket_client.close()
socket_server.close()
