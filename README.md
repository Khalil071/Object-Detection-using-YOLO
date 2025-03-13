Object Detection using YOLO

Overview

This project builds an object detection model using YOLO (You Only Look Once). The model is capable of detecting multiple objects in images with high accuracy and real-time performance. YOLO is a state-of-the-art deep learning model for object detection that balances speed and accuracy.

Features

Detect multiple objects in an image

Use a pre-trained YOLO model (e.g., YOLOv5, YOLOv8)

Perform real-time object detection on images and videos

Visualize detected objects with bounding boxes

Fine-tune the model for custom datasets

Technologies Used

Python

YOLO (You Only Look Once)

OpenCV

PyTorch

TensorFlow/Keras (optional)

Matplotlib/Seaborn

Installation

Clone the repository:

git clone https://github.com/yourusername/object-detection-yolo.git
cd object-detection-yolo

Install dependencies:

pip install -r requirements.txt

Install YOLOv5 (if using YOLOv5):

git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

Model Usage

Load the pre-trained YOLO model:

from ultralytics import YOLO
import cv2

model = YOLO("yolov5s.pt")  # Load the YOLOv5 small model

Detect objects in an image:

image_path = "path/to/image.jpg"
results = model(image_path)
results.show()  # Display results with bounding boxes

Detect objects in a video:

cap = cv2.VideoCapture("path/to/video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    results.show()
cap.release()

Future Improvements

Train YOLO on a custom dataset

Implement real-time object detection using a webcam

Deploy as a web app or API

Optimize the model for edge devices

License

This project is licensed under the MIT License.
