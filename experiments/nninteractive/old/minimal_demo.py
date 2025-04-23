import os
import torch
import SimpleITK as sitk
from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed

# --- Download Trained Model Weights (~400MB) ---
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = "/nfs/norasys/notebooks/camaret/model_checkpoints"  # Specify the download directory
IMAGE_FILENAME = "/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_incontext/amos22/amos22/imagesVa/amos_0311.nii.gz"  # Specify the input image filename
download_path = snapshot_download(
    repo_id=REPO_ID,
    allow_patterns=[f"{MODEL_NAME}/*"],
    local_dir=DOWNLOAD_DIR
)
# The model is now stored in DOWNLOAD_DIR/MODEL_NAME.

# --- Initialize Inference Session ---
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
 
session = nnInteractiveInferenceSession(
    device=torch.device("cuda:0"),  # Set inference device
    use_torch_compile=False,  # Experimental: Not tested yet
    verbose=False,
    torch_n_threads=os.cpu_count(),  # Use available CPU cores
    do_autozoom=True,  # Enables AutoZoom for better patching
    use_pinned_memory=True,  # Optimizes GPU memory transfers
)

# Load the trained model
model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
session.initialize_from_trained_model_folder(model_path)

# --- Load Input Image (Example with SimpleITK) ---
input_image = sitk.ReadImage(IMAGE_FILENAME)
img = sitk.GetArrayFromImage(input_image)[None]  # Ensure shape (1, x, y, z)

# Validate input dimensions
if img.ndim != 4:
    raise ValueError("Input image must be 4D with shape (1, x, y, z)")

session.set_image(img)

# --- Define Output Buffer ---
target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
session.set_target_buffer(target_tensor)

# --- Interacting with the Model ---
# Interactions can be freely chained and mixed in any order. Each interaction refines the segmentation.
# The model updates the segmentation mask in the target buffer after every interaction.

# Example: Add a point interaction
# POINT_COORDINATES should be a tuple (x, y, z) specifying the point location.
POINT_COORDINATES = (50, 50, 10)  # Example point at (50, 50, 10)
session.add_point_interaction(POINT_COORDINATES, include_interaction=True)

# Example: Add a bounding box interaction
# BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
# Note: nnInteractive pre-trained models currently only support **2D bounding boxes**.
# This means that **one dimension must be [d, d+1]** to indicate a single slice.

# Example of a 2D bounding box in the axial plane (XY slice at depth Z)
BBOX_COORDINATES = [[30, 80], [40, 100], [10, 11]]  # X: 30-80, Y: 40-100, Z: slice 10

session.add_bbox_interaction(BBOX_COORDINATES, include_interaction=True)

# Example: Add a scribble interaction
# - A 3D image of the same shape as img where one slice (any axis-aligned orientation) contains a hand-drawn scribble.
# - Background must be 0, and scribble must be 1.
# - Use session.preferred_scribble_thickness for optimal results.
# session.add_scribble_interaction(SCRIBBLE_IMAGE, include_interaction=True)

# Example: Add a lasso interaction
# - Similarly to scribble a 3D image with a single slice containing a **closed contour** representing the selection.
# session.add_lasso_interaction(LASSO_IMAGE, include_interaction=True)

# You can combine any number of interactions as needed. 
# The model refines the segmentation result incrementally with each new interaction.

# --- Retrieve Results ---
# The target buffer holds the segmentation result.
results = session.target_buffer.clone()
# OR (equivalent)
results = target_tensor.clone()

# Cloning is required because the buffer will be **reused** for the next object.
# Alternatively, set a new target buffer for each object:
session.set_target_buffer(torch.zeros(img.shape[1:], dtype=torch.uint8))

# --- Start a New Object Segmentation ---
session.reset_interactions()  # Clears the target buffer and resets interactions

# Now you can start segmenting the next object in the image.

# --- Set a New Image ---
# Setting a new image also requires setting a new matching target buffer
"""
session.set_image(NEW_IMAGE)
session.set_target_buffer(torch.zeros(NEW_IMAGE.shape[1:], dtype=torch.uint8))
"""
# Enjoy!