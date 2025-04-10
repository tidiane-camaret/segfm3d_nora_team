# sam_med3d_predictor_with_helpers_debug.py

import os  # For saving debug files
import sys
import time

import numpy as np
import torch  # Needed by sam_model_infer

# --- Import the provided helper functions ---
# Assume these functions are in the current directory or accessible via PYTHONPATH
try:
    import medim  # For loading the model
    import yaml

    # Assuming config.yaml is in the same directory or parent directory
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        # Try parent directory if not found in current
        config_path = os.path.join("..", "config.yaml")
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
        if "SAM_REPO_DIR" in config and config["SAM_REPO_DIR"] not in sys.path:
            sys.path.append(config["SAM_REPO_DIR"])
        else:
            print("Warning: SAM_REPO_DIR not found in config or already in sys.path.")
    else:
        print(
            "Warning: config.yaml not found. Ensure SAM_REPO_DIR is manually added to PYTHONPATH if needed."
        )

    from medim_infer import (  # Assuming helpers are in this file relative to SAM_REPO_DIR; random_sample_next_click # May not be needed directly if sam_model_infer uses roi_gt
        create_gt_arr,
        data_postprocess,
        data_preprocess,
        sam_model_infer,
    )

    SAM_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import 'medim' or helper functions: {e}")
    print(
        "Ensure 'medim' is installed and the SAM-Med3D repo (containing medim_infer.py) is accessible via PYTHONPATH."
    )
    SAM_AVAILABLE = False
except FileNotFoundError:
    print(
        "Warning: config.yaml specified but not found. Ensure SAM_REPO_DIR is manually added to PYTHONPATH if needed."
    )
    SAM_AVAILABLE = False  # Assume failure if config is needed for path but not found
except KeyError:
    print(
        "Warning: SAM_REPO_DIR key not found in config.yaml. Ensure SAM_REPO_DIR is manually added to PYTHONPATH if needed."
    )
    # SAM_AVAILABLE might still be True if import succeeded without path append


# --- Predictor Class ---


class SAMMed3DPredictor:
    """
    Predictor class for SAM-Med3D using provided helper functions.
    Includes detailed debugging logs and intermediate file saving.
    """

    def __init__(self, checkpoint_path, debug_save_dir="sam_debug_output"):
        """
        Initializes the predictor, loads the model, and sets up debug directory.

        Args:
            checkpoint_path (str): Path to the model checkpoint.
            debug_save_dir (str): Directory to save intermediate numpy arrays.
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM-Med3D components or helpers not found.")

        print(f"Initializing SAMMed3DPredictor with checkpoint: {checkpoint_path}")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")
        self.debug_save_dir = debug_save_dir
        os.makedirs(self.debug_save_dir, exist_ok=True)
        print(f"   Debug outputs will be saved to: {self.debug_save_dir}")

        try:
            # Use medim to load the model
            self.model = medim.create_model(
                "SAM-Med3D",
                # pretrained=True, # Check if needed based on checkpoint
                checkpoint_path=checkpoint_path,
            )
            self.model.to(self.device)  # Move model to device
            self.model.eval()  # Set model to evaluation mode
            print("SAMMed3DPredictor: Model loaded and set to eval mode.")
        except Exception as e:
            print(f"ERROR loading SAM-Med3D model from {checkpoint_path}: {e}")
            raise RuntimeError(
                f"Failed to load SAM-Med3D model from {checkpoint_path}"
            ) from e

    def predict(
        self,
        image_data,  # Full 3D image (NumPy ZYX?)
        spacing_data,  # Voxel spacing (tuple/list XYZ?)
        bbox_data=None,  # BBox list (iter 0) [{'z_min':..,}, ...]
        clicks_data=None,  # Click list (iter 1+) [{'fg':[[x,y,z],..], 'bg':[[x,y,z],..]}, ...]
        prev_pred_data=None,  # Full prediction from previous step (NumPy ZYX?)
        iteration_step=-1,  # Add iteration step for unique debug filenames
        case_name="unknown_case",  # Add case name for unique debug filenames
        num_classes_max=None,  # Optional: limit number of classes processed
    ):
        """
        Performs SAM-Med3D inference using the provided helper functions.
        Includes detailed debugging logs and saves intermediate arrays.
        """
        start_time = time.time()
        # Create a unique prefix for debug files for this specific call
        base_debug_prefix = f"{case_name}_iter{iteration_step}"
        print(
            f"\n--- SAMMed3DPredictor.predict() start (Debug Mode) | Case: {case_name}, Iter: {iteration_step} ---"
        )

        if self.model is None:
            print("   ERROR: Model is not loaded.")
            return np.zeros_like(image_data, dtype=np.uint8), time.time() - start_time

        # Initialize final prediction for THIS iteration step
        final_segmentation = np.zeros_like(image_data, dtype=np.uint8)

        # Ensure prev_pred_data is an array
        if prev_pred_data is None:
            print("   prev_pred_data is None, initializing as zeros.")
            prev_pred_data = np.zeros_like(image_data, dtype=np.uint8)
        else:
            print(
                f"   Received prev_pred_data shape: {prev_pred_data.shape}, dtype: {prev_pred_data.dtype}"
            )
            # Save initial prev_pred_data for this iteration


        # Determine number of classes and iteration type
        num_classes_prompted = 0
        is_bbox_iteration = False
        if bbox_data is not None:
            num_classes_prompted = len(bbox_data)
            is_bbox_iteration = True
            print(f"   Mode: BBox, Classes Prompted: {num_classes_prompted}")
        elif clicks_data is not None:
            num_classes_prompted = len(clicks_data)
            print(f"   Mode: Clicks, Classes Prompted: {num_classes_prompted}")
        else:
            print("   Warning: No prompts provided. Returning previous prediction.")
            return prev_pred_data.copy(), time.time() - start_time

        # Determine number of classes to process
        num_classes = (
            num_classes_prompted
            if num_classes_max is None
            else min(num_classes_max, num_classes_prompted)
        )
        print(f"   Processing up to {num_classes} classes.")

        # --- Iterate through classes ---
        for idx in range(num_classes):
            class_index = idx + 1  # SAM uses 1-based indexing
            # Unique prefix for files related to this class in this iteration
            debug_prefix = f"{base_debug_prefix}_class{class_index}"
            print(f"\n--- Processing Class Index: {class_index} ---")

            # --- Verify Inputs for this Class ---
            cls_prev_seg_mask = (prev_pred_data == class_index).astype(np.uint8)
            print(
                f"   Input cls_prev_seg_mask shape: {cls_prev_seg_mask.shape}, Sum: {np.sum(cls_prev_seg_mask)}"
            )
            np.save(
                os.path.join(
                    self.debug_save_dir, f"{debug_prefix}_0b_cls_prev_seg_mask.npy"
                ),
                cls_prev_seg_mask,
            )

            # --- Copy previous prediction to output (will be overwritten if inference succeeds) ---
            final_segmentation[cls_prev_seg_mask != 0] = class_index
            print(
                f"   Copied previous prediction area for class {class_index} to initial output."
            )

            # --- Determine and Verify Prompt Point ---
            prompt_point = None  # Expected format ZYX for create_gt_arr
            is_positive_prompt = False

            if is_bbox_iteration:
                bbox = bbox_data[idx]
                z_cen = (bbox["z_min"] + bbox["z_max"]) / 2
                y_cen = (bbox["z_mid_y_min"] + bbox["z_mid_y_max"]) / 2
                x_cen = (bbox["z_mid_x_min"] + bbox["z_mid_x_max"]) / 2
                prompt_point = (z_cen, y_cen, x_cen)  # Assuming ZYX order
                is_positive_prompt = True
                print(f"   Using BBox center as prompt point (ZYX?): {prompt_point}")
            else:  # Click iteration
                fg_clicks = clicks_data[idx].get(
                    "fg", []
                )  # Expected format: list of [x,y,z]
                bg_clicks = clicks_data[idx].get(
                    "bg", []
                )  # Expected format: list of [x,y,z]
                if fg_clicks:
                    last_click_xyz = fg_clicks[-1]
                    prompt_point = (
                        last_click_xyz[2],
                        last_click_xyz[1],
                        last_click_xyz[0],
                    )  # Convert XYZ to ZYX
                    is_positive_prompt = True
                    print(
                        f"   Using last FG click (XYZ: {last_click_xyz}) as prompt point (ZYX): {prompt_point}"
                    )
                elif bg_clicks:
                    last_click_xyz = bg_clicks[-1]
                    prompt_point = (
                        last_click_xyz[2],
                        last_click_xyz[1],
                        last_click_xyz[0],
                    )  # Convert XYZ to ZYX
                    is_positive_prompt = False
                    print(
                        f"   Using last BG click (XYZ: {last_click_xyz}) as prompt point (ZYX): {prompt_point}"
                    )
                else:
                    print(f"   Skipping class {class_index}: No clicks in this step.")
                    continue

            print(
                f"   Prompt point determined: {prompt_point}, is_positive: {is_positive_prompt}"
            )

            # --- Mimic condition: Only proceed if positive prompt ---
            if not is_positive_prompt:
                print(f"   Skipping SAM steps: Last prompt was negative.")
                continue

            if prompt_point is None:
                print(f"   ERROR: Prompt point is None unexpectedly. Skipping.")
                continue

            # --- Generate and Verify Prompt Mask (cls_gt) ---
            try:
                # Ensure coords are int and within bounds for indexing
                int_prompt_point = tuple(
                    int(max(0, min(image_data.shape[d] - 1, prompt_point[d])))
                    for d in range(3)
                )
                print(
                    f"   Integer prompt point for create_gt_arr (ZYX): {int_prompt_point}"
                )

                cls_gt = create_gt_arr(
                    image_data.shape, int_prompt_point, category_index=class_index
                )

                print(
                    f"   Generated cls_gt shape: {cls_gt.shape}, Sum: {np.sum(cls_gt)}, Unique values: {np.unique(cls_gt)}"
                )
                np.save(
                    os.path.join(self.debug_save_dir, f"{debug_prefix}_1_cls_gt.npy"),
                    cls_gt,
                )
                print(f"   Saved {debug_prefix}_1_cls_gt.npy")
                if np.sum(cls_gt) == 0:
                    print(
                        f"   !!! CRITICAL WARNING: cls_gt is empty! Check create_gt_arr or prompt_point conversion."
                    )
                    continue

            except Exception as e:
                print(f"   ERROR generating cls_gt prompt for class {class_index}: {e}")
                import traceback

                traceback.print_exc()
                continue

            # --- Run SAM Steps using Helpers ---
            try:
                # 1. Preprocess
                print(f"   Preprocessing...")
                print(
                    f"     Input image_data shape: {image_data.shape}, dtype: {image_data.dtype}"
                )
                print(
                    f"     Input cls_gt shape: {cls_gt.shape}, dtype: {cls_gt.dtype}, sum: {np.sum(cls_gt)}"
                )
                print(
                    f"     Input cls_prev_seg_mask shape: {cls_prev_seg_mask.shape}, dtype: {cls_prev_seg_mask.dtype}"
                )
                print(f"     Input spacing_data (XYZ?): {spacing_data}")

                # --- Check/Adjust spacing order for data_preprocess ---
                # Assuming data_preprocess uses torchio Resample which likely expects XYZ target spacing if zipped with XYZ orig_spacing
                current_spacing = [spacing_data[2], spacing_data[0], spacing_data[1]]

                roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(
                    image_data.astype(
                        np.float32
                    ),  # Ensure float input for normalization
                    cls_gt,
                    cls_prev_seg_mask,
                    orig_spacing=current_spacing,
                    category_index=class_index,
                )
                print(f"   Preprocessing done.")
                print(
                    f"     roi_image shape: {roi_image.shape}, dtype: {roi_image.dtype}"
                )  # Expected: [1, 1, D, H, W] Tensor
                print(
                    f"     roi_label shape: {roi_label.shape}, sum: {roi_label.sum()}, dtype: {roi_label.dtype}"
                )  # Prompt mask in ROI
                print(
                    f"     roi_prev_seg shape: {roi_prev_seg.shape}, sum: {roi_prev_seg.sum()}, dtype: {roi_prev_seg.dtype}"
                )  # Prev seg in ROI
                print(f"     meta_info: {meta_info}")
                # Save ROI outputs
                np.save(
                    os.path.join(
                        self.debug_save_dir, f"{debug_prefix}_2_roi_image.npy"
                    ),
                    roi_image.cpu().numpy(),
                )
                np.save(
                    os.path.join(
                        self.debug_save_dir, f"{debug_prefix}_2_roi_label.npy"
                    ),
                    roi_label.cpu().numpy(),
                )
                np.save(
                    os.path.join(
                        self.debug_save_dir, f"{debug_prefix}_2_roi_prev_seg.npy"
                    ),
                    roi_prev_seg.cpu().numpy(),
                )
                print(f"   Saved ROI arrays for class {class_index}")

                # 2. Inference (CRITICAL STEP)
                print(f"   Running SAM inference...")
                print(f"     Inputs to sam_model_infer:")
                print(f"       roi_image shape: {roi_image.shape}")
                print(
                    f"       roi_gt (roi_label) shape: {roi_label.shape}, sum: {roi_label.sum()}"
                )
                print(
                    f"       prev_low_res_mask (roi_prev_seg) shape: {roi_prev_seg.shape}, sum: {roi_prev_seg.sum()}"
                )

                # Ensure inputs are on the correct device if model is on GPU
                roi_image = roi_image.to(self.device)
                roi_label = roi_label.to(self.device)
                roi_prev_seg = roi_prev_seg.to(self.device)

                roi_pred = sam_model_infer(
                    self.model,
                    roi_image,
                    roi_gt=roi_label,  # Pass the processed prompt mask
                    prev_low_res_mask=roi_prev_seg,  # Pass the previous mask in ROI space
                )
                # roi_pred is expected: NumPy array [D, H, W] (binary 0/1)
                print(f"   Inference done.")
                print(
                    f"     roi_pred output shape: {roi_pred.shape}, dtype: {roi_pred.dtype}, Sum: {np.sum(roi_pred)}, Unique: {np.unique(roi_pred)}"
                )
                np.save(
                    os.path.join(self.debug_save_dir, f"{debug_prefix}_3_roi_pred.npy"),
                    roi_pred,
                )
                print(f"   Saved {debug_prefix}_3_roi_pred.npy")

                if np.sum(roi_pred) == 0:
                    print(
                        f"   !!! CRITICAL WARNING: roi_pred is empty! Inference failed or produced no segmentation."
                    )
                    # continue # Optionally skip postprocessing if prediction is empty

                # 3. Postprocess
                print(f"   Postprocessing...")
                # data_postprocess expects roi_pred as numpy [D,H,W]
                pred_ori = data_postprocess(roi_pred, meta_info, output_dir=None)
                # pred_ori should be NumPy array in original image space (ZYX?)
                print(f"   Postprocessing done.")
                print(
                    f"     pred_ori shape: {pred_ori.shape}, dtype: {pred_ori.dtype}, Sum: {np.sum(pred_ori)}, Unique: {np.unique(pred_ori)}"
                )
                np.save(
                    os.path.join(self.debug_save_dir, f"{debug_prefix}_4_pred_ori.npy"),
                    pred_ori,
                )
                print(f"   Saved {debug_prefix}_4_pred_ori.npy")

                # 4. Aggregate Result
                update_mask = pred_ori != 0
                sum_before = np.sum(final_segmentation == class_index)
                final_segmentation[update_mask] = class_index
                sum_after = np.sum(final_segmentation == class_index)
                print(
                    f"   Aggregation: Sum before={sum_before}, Sum after={sum_after}. Updated voxels: {np.sum(update_mask)}"
                )

            except Exception as e:
                print(f"   ERROR during SAM steps for class {class_index}: {e}")
                import traceback

                traceback.print_exc()
                print(
                    f"   Keeping previous prediction for class {class_index} due to error."
                )
                # The copied previous prediction remains in final_segmentation

        # --- End of class loop ---

        # Save final segmentation for this iteration step
        final_segmentation_path = os.path.join(
            self.debug_save_dir, f"{base_debug_prefix}_5_final_segmentation.npy"
        )
        np.save(final_segmentation_path, final_segmentation)
        print(f"Saved final segmentation to {final_segmentation_path}")

        inference_time = time.time() - start_time
        print(f"--- SAMMed3DPredictor.predict() end. Time: {inference_time:.2f}s ---")

        return final_segmentation, inference_time
