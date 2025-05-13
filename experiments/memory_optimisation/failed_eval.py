import os
from src.config import config

method = "nnint"
img_dir = os.path.join(config["DATA_DIR"], "3D_val_npz")
gt_dir = os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg")
output_dir = os.path.join(config["RESULTS_DIR"], method)# difference of files in the two directories
img_files = set(os.listdir(img_dir))
gt_files = set(os.listdir(gt_dir)) # 4 files from gt are missing from img
output_files = set(os.listdir(output_dir))
print("img_files", len(img_files))
print("output_files", len(output_files))
