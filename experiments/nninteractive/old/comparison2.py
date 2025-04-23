import os
import numpy as np
import torch
import SimpleITK as sitk
import onnxruntime as ort
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd
import nibabel as nib
from datetime import datetime
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Define paths
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = "/nfs/norasys/notebooks/camaret/model_checkpoints"
IMAGE_FILENAME = "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/amos22/amos22/imagesVa/amos_0311.nii.gz"
ONNX_PATH = os.path.join(DOWNLOAD_DIR, f"{MODEL_NAME}.onnx")
OUTPUT_DIR = "/nfs/norasys/notebooks/camaret/comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import os
import torch
import torch.nn as nn
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

# Path to the downloaded model
MODEL_PATH = "/nfs/norasys/notebooks/camaret/model_checkpoints/nnInteractive_v1.0"

# Initialize session
pytorch_session = nnInteractiveInferenceSession(
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_torch_compile=False,
    verbose=True,  # Enable verbose mode for insights
    torch_n_threads=os.cpu_count(),
    do_autozoom=True,
    use_pinned_memory=True
)
pytorch_session.initialize_from_trained_model_folder(MODEL_PATH)


pytorch_session.network.eval()
# Make sure all submodules are also in eval mode
def set_all_to_eval(module):
    for module in module.modules():
        module.train(False)
           
        

set_all_to_eval(pytorch_session.network)

# Create a dummy batch with the expected input format
# The model expects an input with shape [batch_size, channels=8, D, H, W]
# where channels are: [image, initial_seg, bbox+, bbox-, point+, point-, scribble+, scribble-]
dummy_shape = (1, 8, 64, 128, 128)  # batch_size, channels, depth, height, width
dummy_input = torch.randn(dummy_shape, dtype=torch.float32).to("cuda:0")  # Using random values instead of zeros

print(f"Exporting model with input shape: {dummy_input.shape}")

# Export the model
try:
    torch.onnx.export(
        pytorch_session.network,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['combined_input'],
        output_names=['segmentation'],
        dynamic_axes={
            'combined_input': {2: 'depth', 3: 'height', 4: 'width'},
            'segmentation': {2: 'depth', 3: 'height', 4: 'width'}
        }
    )
    print(f"Model exported successfully to {ONNX_PATH}")
except Exception as e:
    print(f"Error exporting model: {e}")


# --- PyTorch Pipeline (Original) ---
print("\n--- Running PyTorch Pipeline ---")
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession



# Set the image for the PyTorch session
pytorch_output = pytorch_session.network(dummy_input).detach().cpu().numpy()

print("pytorch output shape : ", pytorch_output.shape)

# Run ONNX inference
onnx_session = ort.InferenceSession(ONNX_PATH,
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# Create input tensor for ONNX with correct channel structure
onnx_dummy_input = dummy_input.detach().cpu().numpy()
print("Running ONNX inference...")
onnx_output = onnx_session.run(None, {'combined_input': onnx_dummy_input})
print("onnx output len : ", len(onnx_output))

onnx_output_0 = onnx_output[0] 

print("onnx output first elt shape : ", onnx_output_0.shape)
 # Shape: (1, 2, 64, 128, 128)
diff = pytorch_output - onnx_output_0
print(f"Max diff: {np.max(np.abs(diff))}, Mean diff: {np.mean(np.abs(diff))}")
print(f"Are outputs equal: {np.allclose(pytorch_output, onnx_output_0, rtol=1e-4, atol=1e-4)}")

