# validate_onnx.py
import numpy as np
import onnxruntime as ort
import torch

# Load the ONNX model
ONNX_OUTPUT_PATH = "/nfs/norasys/notebooks/camaret/model_checkpoints/nnInteractive_v1.0.onnx"

session = ort.InferenceSession(ONNX_OUTPUT_PATH)

# Create test input
test_input = np.zeros((1, 8, 64, 128, 128), dtype=np.float32)
# Add a point interaction
test_input[0, 4, 32, 64, 32] = 1.0  # point+ channel (index 4)

# Run inference
outputs = session.run(None, {'combined_input': test_input})
print(f"Output shape: {outputs[0].shape}")