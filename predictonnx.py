import numpy as np
import onnxruntime as rt

import matplotlib.pyplot as plt
# Load the input image and preprocess it
img =open('C:\\Users\\User\\OneDrive\\Desktop\\YOLODataset\\1.png')
img_resized = img.resize(640, 640)
img_resized_arr = np.array(img_resized)
img_resized_arr = np.expand_dims(img_resized_arr, axis=0)
img_resized_arr = np.transpose(img_resized_arr, (0, 3, 1, 2))
img_resized_arr = img_resized_arr.astype(np.float32)

# Load the ONNX model and get the input and output names
sess = rt.InferenceSession('C:\\Users\\User\\OneDrive\\Desktop\\dataset\\best.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Perform inference on the input data
output = sess.run([output_name], {input_name: img_resized_arr})[0]

# Postprocess the output tensor to obtain the final predictions
class_idx = np.argmax(output, axis=1)
class_prob = output[0, class_idx]

# Print the predicted class index and probability
print('Predicted class index:', class_idx)
print('Predicted class probability:', class_prob)
# Create a color map for the classes
cmap = plt.get_cmap('tab10')
# Plot the class indices as an image
plt.imshow(class_idx, cmap=cmap, vmin=0, vmax=9)

# Add color bar legend for classes
cbar = plt.colorbar()
cbar.set_ticks(np.arange(10))
cbar.set_ticklabels(['Class {}'.format(i) for i in range(10)])

# Show the plot
plt.show()

# Load the ground truth labels
with open('C:\\Users\\User\\OneDrive\\Desktop\\dataaaa\\test\\labels', 'r') as f:
    labels = [line.strip() for line in f]

# Get the predicted class index
class_idx = np.argmax(output, axis=1)[0]

# Get the predicted class label
class_label = labels[class_idx]

# Print the predicted class label
print('Predicted label:', class_label)

# Load the input image
img2 = ('C:\\Users\\User\\OneDrive\\Desktop\\dataaaa\\val\\images\\1 (27).png')

# Plot the input image
plt.imshow(img2)

# Add the predicted label as the title
plt.title(class_label)

# Show the plot
plt.show()
