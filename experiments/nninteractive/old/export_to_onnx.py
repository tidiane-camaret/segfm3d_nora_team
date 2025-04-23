import os
import torch
import torch.nn as nn
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

# Path to the downloaded model
MODEL_PATH = "/nfs/norasys/notebooks/camaret/model_checkpoints/nnInteractive_v1.0"
ONNX_OUTPUT_PATH = "/nfs/norasys/notebooks/camaret/model_checkpoints/nnInteractive_v1.0.onnx"

# Initialize session
session = nnInteractiveInferenceSession(device=torch.device("cuda:0"))
session.initialize_from_trained_model_folder(MODEL_PATH)

print("Session initialized successfully")

# Create a wrapper class that exposes the network in a way suitable for ONNX export
class ONNXExportWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
    
    def forward(self, combined_input):
        """
        The model expects a combined input with:
        - First channel: image data
        - Remaining channels: interaction data (initial seg, bbox+, bbox-, point+, point-, scribble+, scribble-)
        """
        return self.network(combined_input)

# Create the wrapper around the actual network
model_wrapper = ONNXExportWrapper(session.network)

# Create a dummy batch with the expected input format
# The model expects an input with shape [batch_size, channels=8, D, H, W]
# where channels are: [image, initial_seg, bbox+, bbox-, point+, point-, scribble+, scribble-]
dummy_shape = (1, 8, 64, 128, 128)  # batch_size, channels, depth, height, width
dummy_input = torch.zeros(dummy_shape, dtype=torch.float32).to("cuda:0")

print(f"Exporting model with input shape: {dummy_input.shape}")

# Export the model
try:
    torch.onnx.export(
        model_wrapper,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['combined_input'],
        output_names=['segmentation'],
        dynamic_axes={
            'combined_input': {2: 'depth', 3: 'height', 4: 'width'},
            'segmentation': {2: 'depth', 3: 'height', 4: 'width'}
        }
    )
    print(f"Model exported successfully to {ONNX_OUTPUT_PATH}")
except Exception as e:
    print(f"Error exporting model: {e}")