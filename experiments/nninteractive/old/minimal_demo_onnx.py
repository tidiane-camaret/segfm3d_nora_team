import numpy as np
import onnxruntime as ort
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Load the ONNX model
ONNX_OUTPUT_PATH = "/nfs/norasys/notebooks/camaret/model_checkpoints/nnInteractive_v1.0.onnx"
IMAGE_FILENAME = "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/amos22/amos22/imagesVa/amos_0311.nii.gz"

# Initialize ONNX session
session = ort.InferenceSession(ONNX_OUTPUT_PATH)

# --- Load and preprocess the image (replicating nnInteractive's internal processing) ---
def preprocess_image(filename, target_shape=(64, 128, 128)):
    # Load image using SimpleITK (just like in the original)
    input_image = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(input_image)
    
    # Add batch dimension if not present
    if img_array.ndim == 3:
        img_array = img_array[np.newaxis, ...]
    
    print(f"Original image shape: {img_array.shape}")
    
    # Resize to target shape (replicating 'do_autozoom')
    factors = [t/s for t, s in zip(target_shape, img_array.shape[1:])]
    resized_img = zoom(img_array, [1] + factors, order=1)
    print(f"Resized image shape: {resized_img.shape}")
    
    # Normalize intensity values (0-1 range)
    img_min, img_max = resized_img.min(), resized_img.max()
    if img_max > img_min:
        normalized_img = (resized_img - img_min) / (img_max - img_min)
    else:
        normalized_img = resized_img
    
    return normalized_img.astype(np.float32)

# --- Create multi-channel input tensor (replicating internal nnInteractive structure) ---
def create_input_tensor(img, interactions=None):
    # Create 8-channel tensor expected by the model
    input_tensor = np.zeros((1, 8, img.shape[1], img.shape[2], img.shape[3]), dtype=np.float32)
    
    # Set image data in channel 0
    input_tensor[0, 0] = img[0]
    
    # Add interactions if provided
    if interactions:
        for interaction_type, coords in interactions:
            if interaction_type == "point+":
                # Points are placed in channel 4 for positive points
                z, y, x = coords  # Adjust based on your coordinate system
                input_tensor[0, 4, z, y, x] = 1.0
            elif interaction_type == "point-":
                # Negative points would go in channel 5
                z, y, x = coords
                input_tensor[0, 5, z, y, x] = 1.0
            # Additional interaction types would be handled similarly
    
    return input_tensor

# --- Process model output (replicating nnInteractive's post-processing) ---
def postprocess_output(output):
    # Convert from probability maps to binary segmentation
    # Typically done with argmax for binary segmentation
    segmentation = np.argmax(output[0], axis=0)
    return segmentation

# Run the pipeline
img = preprocess_image(IMAGE_FILENAME)

# Define interactions (equivalent to POINT_COORDINATES in the original)
interactions = [
    ("point+", (30, 64, 64)),  # Format: (z, y, x)
]

# Create input tensor with image and interactions
input_tensor = create_input_tensor(img, interactions)

# Run inference
outputs = session.run(None, {'combined_input': input_tensor})

# Process output
segmentation = postprocess_output(outputs[0])

# Visualize results
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(131)
plt.title("Original Image (middle slice)")
plt.imshow(img[0, img.shape[1]//2])

# Segmentation result
plt.subplot(132)
plt.title("Segmentation Result (middle slice)")
plt.imshow(segmentation[img.shape[1]//2])

# Overlay
plt.subplot(133)
plt.title("Overlay")
overlay = img[0, img.shape[1]//2].copy()
overlay[segmentation[img.shape[1]//2] > 0] = 1
plt.imshow(overlay)

plt.tight_layout()
plt.show()