#!/usr/bin/env python
# Compare PyTorch and ONNX implementations with exact preprocessing

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

# Define paths
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = "/nfs/norasys/notebooks/camaret/model_checkpoints"
IMAGE_FILENAME = "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/amos22/amos22/imagesVa/amos_0311.nii.gz"
ONNX_PATH = os.path.join(DOWNLOAD_DIR, f"{MODEL_NAME}.onnx")
OUTPUT_DIR = "/nfs/norasys/notebooks/camaret/comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, f"exact_compare_{timestamp}")
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

print(f"Results will be saved to: {OUTPUT_SUBDIR}")

# --- PyTorch Pipeline (Original) ---
print("\n--- Running PyTorch Pipeline ---")
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

pytorch_session = nnInteractiveInferenceSession(
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_torch_compile=False,
    verbose=True,  # Enable verbose mode for insights
    torch_n_threads=os.cpu_count(),
    do_autozoom=True,
    use_pinned_memory=True
)

# Load the trained model
model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
pytorch_session.initialize_from_trained_model_folder(model_path)

# Load the input image
input_image_sitk = sitk.ReadImage(IMAGE_FILENAME)
input_image_orig = sitk.GetArrayFromImage(input_image_sitk)[None]  # Shape: (1, Z, Y, X)
original_shape = input_image_orig.shape
print(f"Original image shape: {original_shape}")

# Set the image for the PyTorch session
pytorch_session.set_image(input_image_orig)

# Define target buffer
target_tensor = torch.zeros(input_image_orig.shape[1:], dtype=torch.uint8)
pytorch_session.set_target_buffer(target_tensor)

# Define interaction points - use middle of image
center_point = (input_image_orig.shape[1] // 2, input_image_orig.shape[2] // 2, input_image_orig.shape[3] // 2)
print(f"Using center point: {center_point}")

# Apply interaction to PyTorch pipeline
pytorch_session.add_point_interaction(center_point, include_interaction=True)

# Get PyTorch results
pytorch_result = pytorch_session.target_buffer.detach().cpu().numpy()
print(f"PyTorch result shape: {pytorch_result.shape}, unique values: {np.unique(pytorch_result)}")

# --- Extract the exact preprocessing done by PyTorch ---
# This is crucial - we need to extract what preprocessing was actually done
print("\n--- Extracting PyTorch Preprocessing Details ---")

# Get the preprocessing properties
bbox_used_for_cropping = pytorch_session.preprocessed_props['bbox_used_for_cropping']
print(f"Bounding box used for cropping: {bbox_used_for_cropping}")

# Get the preprocessed image and interactions
preprocessed_image = pytorch_session.preprocessed_image.cpu().numpy()
print(f"Preprocessed image shape: {preprocessed_image.shape}")
print(f"Preprocessed image range: [{preprocessed_image.min():.4f}, {preprocessed_image.max():.4f}], mean: {preprocessed_image.mean():.4f}, std: {preprocessed_image.std():.4f}")

# --- ONNX Pipeline with Exact PyTorch Preprocessing ---
print("\n--- Running ONNX Pipeline with Exact PyTorch Preprocessing ---")

# Initialize ONNX session
onnx_session = ort.InferenceSession(ONNX_PATH)

# Exactly replicate PyTorch preprocessing
def replicate_pytorch_preprocessing(image):
    """Replicate the exact preprocessing done by PyTorch"""
    # Convert to tensor
    image_torch = torch.from_numpy(image).clone()
    
    # Crop to nonzero region - exactly as in _background_set_image
    print("Cropping input image to nonzero region")
    nonzero_idx = torch.where(image_torch != 0)
    # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices
    bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
    slicer = bounding_box_to_slice(bbox)
    image_torch = image_torch[slicer].float()
    print(f"Cropped image shape: {image_torch.shape}")
    
    # Normalize the cropped image - exactly as in _background_set_image
    print("Normalizing cropped image")
    image_torch -= image_torch.mean()
    image_torch /= image_torch.std()
    
    return image_torch.numpy(), bbox

# Process the image exactly like PyTorch
preprocessed_img, bbox_info = replicate_pytorch_preprocessing(input_image_orig)
print(f"Preprocessed image shape: {preprocessed_img.shape}")

# Create ONNX input tensor with proper dimensions
target_shape = (64, 128, 128)

# Fixed resizing function that properly handles dimensions
def resize_to_target_shape(img_array, target_shape):
    """Resize an image array to target shape (correctly handling dimensions)"""
    # Get input shape (ignoring batch dimension)
    input_shape = img_array.shape[1:]
    
    # Calculate factors for each spatial dimension
    factors = [float(t) / float(s) for t, s in zip(target_shape, input_shape)]
    print(f"Resize factors: {factors}")
    
    # Apply zoom with correct order of dimensions
    # For order=1, this is equivalent to linear interpolation
    resized = zoom(img_array[0], factors, order=1)
    
    print(f"Resized shape: {resized.shape}")
    return resized

# Resize preprocessed image correctly
resized_img = resize_to_target_shape(preprocessed_img, target_shape)

# Create input tensor for ONNX with correct channel structure
onnx_input = np.zeros((1, 8, 64, 128, 128), dtype=np.float32)

# Set image in channel 0 (ONNX) - this should now have the right dimensions
onnx_input[0, 0, :, :, :] = resized_img

# Transform the interaction point from original space to preprocessed space
def transform_coordinates(coords_orig, crop_bbox, target_shape, orig_shape):
    """Transform coordinates from original image to the resized space"""
    # First, transform to cropped space (as in transform_coordinates_noresampling)
    cropped_coords = [coords_orig[d] - crop_bbox[d][0] for d in range(len(coords_orig))]
    
    # Then, transform to resized space
    resize_factors = [float(t) / float(s) for t, s in zip(target_shape, orig_shape[1:])]
    resized_coords = [int(c * f) for c, f in zip(cropped_coords, resize_factors)]
    
    return tuple(resized_coords)

# Map the center point to the preprocessed and resized space
transformed_point = transform_coordinates(
    center_point, 
    bbox_used_for_cropping, 
    target_shape, 
    original_shape
)
print(f"Transformed interaction point: {center_point} -> {transformed_point}")

# Set the point interaction in channel 4 (positive points)
# Using a small radius to mimic the PointInteraction_stub behavior
point_radius = 1  # Default radius in the PyTorch implementation
print(f"Using point radius: {point_radius}")

# Function to place a point with a radius (similar to PointInteraction_stub)
def place_point_with_radius(tensor, coords, radius):
    """Place a point with a given radius in the tensor"""
    z, y, x = coords
    # Get tensor shape
    depth, height, width = tensor.shape
    
    # Create coordinates where the point should be placed (within bounds)
    z_coords = np.arange(max(0, z-radius), min(depth, z+radius+1))
    y_coords = np.arange(max(0, y-radius), min(height, y+radius+1))
    x_coords = np.arange(max(0, x-radius), min(width, x+radius+1))
    
    # Create meshgrid
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Calculate distances
    distances = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
    
    # Set values where distance <= radius
    for i in range(len(z_coords)):
        for j in range(len(y_coords)):
            for k in range(len(x_coords)):
                if distances[i, j, k] <= radius:
                    tensor[z_coords[i], y_coords[j], x_coords[k]] = 1.0
                    
    return tensor

# Place the point in channel 4
onnx_input[0, 4] = place_point_with_radius(
    np.zeros(target_shape, dtype=np.float32), 
    transformed_point, 
    point_radius
)

# Run ONNX inference
print("Running ONNX inference...")
onnx_outputs = onnx_session.run(None, {'combined_input': onnx_input})
onnx_result_raw = onnx_outputs[0][0]  # Shape: (2, 64, 128, 128)

# Convert to segmentation mask
onnx_result = np.argmax(onnx_result_raw, axis=0)  # Shape: (64, 128, 128)
print(f"ONNX result shape: {onnx_result.shape}, unique values: {np.unique(onnx_result)}")

# Save the ONNX output as is, for inspection
onnx_direct_output = os.path.join(OUTPUT_SUBDIR, "onnx_direct_output.npy")
np.save(onnx_direct_output, onnx_result)

# Save normalized image for inspection
np.save(os.path.join(OUTPUT_SUBDIR, "normalized_resized_input.npy"), resized_img)

# --- Save and compare results ---
print("\nSaving and comparing results...")

# Save results as NIfTI
def save_nifti(data, filename, reference_image=None):
    """Save data as NIfTI file"""
    if reference_image:
        # Use reference image geometry
        img = reference_image.__class__(data.astype(np.float32), reference_image.affine)
    else:
        img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, filename)
    print(f"Saved: {filename}")

# Load reference image
reference_img = nib.load(IMAGE_FILENAME)

# Save PyTorch result
save_nifti(pytorch_result, os.path.join(OUTPUT_SUBDIR, "pytorch_result.nii.gz"), reference_img)

# Since we can't easily transform the ONNX result back to original space,
# save it as-is for now and we can compare visually
save_nifti(onnx_result, os.path.join(OUTPUT_SUBDIR, "onnx_result_native_space.nii.gz"))

# Create visualizations of ONNX result
def save_onnx_visualization(onnx_result, output_path, title="ONNX Result"):
    plt.figure(figsize=(15, 10))
    n_rows, n_cols = 4, 4
    slices = np.linspace(0, onnx_result.shape[0]-1, n_rows*n_cols).astype(int)
    
    for i, slice_idx in enumerate(slices):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(onnx_result[slice_idx], cmap='viridis')
        plt.title(f"Slice {slice_idx}")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

save_onnx_visualization(onnx_result, os.path.join(OUTPUT_SUBDIR, "onnx_result_visualization.png"))

# Save input visualization for comparison
save_onnx_visualization(resized_img, os.path.join(OUTPUT_SUBDIR, "input_visualization.png"), 
                       "Preprocessed Input (Normalized & Resized)")

# Save detailed comparison analysis
stats_file = os.path.join(OUTPUT_SUBDIR, "analysis_details.txt")
with open(stats_file, 'w') as f:
    f.write("PyTorch vs ONNX Preprocessing Analysis\n")
    f.write("=====================================\n\n")
    f.write(f"Original image shape: {original_shape}\n")
    f.write(f"PyTorch preprocessed shape: {preprocessed_image.shape}\n")
    f.write(f"ONNX input shape: {onnx_input.shape}\n")
    f.write(f"Bounding box used for cropping: {bbox_used_for_cropping}\n")
    f.write(f"Point interaction at: {center_point}\n")
    f.write(f"Transformed point: {transformed_point}\n\n")
    
    f.write("PyTorch preprocessing steps:\n")
    f.write("1. Crop to non-zero region\n")
    f.write("2. Normalize by subtracting mean and dividing by std\n")
    f.write("3. Apply interactions\n")
    f.write("4. Auto-zoom for patching\n\n")
    
    f.write("ONNX preprocessing steps:\n")
    f.write("1. Same crop and normalization\n")
    f.write("2. Resize to fixed (64, 128, 128)\n")
    f.write("3. Transform interaction points to this space\n\n")
    
    f.write("Differences noted:\n")
    f.write("- PyTorch pipeline uses auto-zoom for adaptive refinement\n")
    f.write("- ONNX requires fixed input dimensions\n")
    f.write("- PyTorch can handle arbitrary resolution original images\n")

print(f"\nComparison complete. Results saved to {OUTPUT_SUBDIR}")