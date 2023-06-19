import torch
import onnx

# Load the PyTorch model
model = torch.load('C:\\Users\\User\\OneDrive\\Desktop\\best.pt')

# Export the model to ONNX format
input_shape = (1, 3, 640, 640)
input_names = ["X"]
output_names = ["Y"]
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
dummy_input = torch.randn(input_shape)
torch.onnx.export(model, dummy_input, "C:\\Users\\User\\Downloads\\onnx", input_names=input_names,
                  output_names=output_names, dynamic_axes=dynamic_axes)

# Save the ONNX model to a file
onnx.save(model, "C:\\Users\\User\\Downloads\\onnx")
