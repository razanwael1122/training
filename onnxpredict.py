'''
from ultralytics import YOLO 
model = YOLO("C:\\Users\\User\\OneDrive\\Desktop\\dataset\\best.onnx")
results = model.predict("C:\\Users\\User\\OneDrive\\Desktop\\YOLODataset\\1.png", show=True, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_crop=False, line_thickness=2)
'''


import onnxruntime as ort
import cv2
import numpy as np
# Load the ONNX model
model = ort.InferenceSession("C:\\Users\\User\\OneDrive\\Desktop\\dataset\\best.onnx")

# Define the input and output names of the model
input_name = model.get_inputs()[0].name
output_names = [output.name for output in model.get_outputs()]

# Load the input image
image = cv2.imread("C:\\Users\\User\\OneDrive\\Desktop\\YOLODataset\\1.png")

# Preprocess the image

image = image.astype(np.float32)# Normalize pixel values to [0, 1]

# Run the model on the input image

image = np.expand_dims(image, axis=0)

outputs = model.run(output_names, {input_name: image})
# Process the model outputs to extract bounding boxes and class labels
# The output format may vary depending on the model architecture and ONNX export settings
# You may need to consult the model documentation or experiment with different output formats
boxes = outputs[0]
scores = outputs[1]
labels = outputs[2]

# Print the results
for box, score, label in zip(boxes, scores, labels):
    print(f"Class {label}: {score:.2f} - Bounding box: {box}")