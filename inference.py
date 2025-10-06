import onnxruntime as ort
import numpy as np
import os
import cv2

# ===========================
# Configuration
# ===========================
base_path = r"D:\braineye"
dataset_path = os.path.join(base_path, "dataset", "dataset")
model_path = os.path.join(base_path, "models")



# ===========================
# Select a random image from the dataset for inference
# ===========================
test_folder = os.path.join(dataset_path, "20")
files = os.listdir(test_folder)
# randomly select one file
test_image = os.path.join(test_folder, np.random.choice(files))


# ===========================
# Load the image and preprocess
# ===========================
image = cv2.imread(test_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Preprocessing of the image.
image = cv2.resize(image, (224, 224))
#normalize the image
image = (image / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
image = np.array(image).transpose(2, 0, 1)  # HWC to CHW
image = np.expand_dims(image, axis=0)  # Add batch dimension

# ===========================
# Load the ONNX model.
# ===========================
onnx_model_path = os.path.join(model_path, "age_prediction_0.0.1.onnx")


# Create ONNX Runtime session
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Prepare input for ONNX Runtime 
onnx_input = {ort_session.get_inputs()[0].name: image.astype(np.float32)}

# Run inference
onnx_output = ort_session.run(None, onnx_input)
onnx_output = onnx_output[0]

print(f"Predicted age: {onnx_output.squeeze()}")
