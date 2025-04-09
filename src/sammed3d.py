# sam_med3d_predictor_with_helpers.py

import numpy as np
import time
import sys
import torch # Needed by sam_model_infer

# --- Import the provided helper functions ---
# Assume these functions are in the current directory or accessible via PYTHONPATH
try:
    import medim # For loading the model
    import yaml
    config = yaml.safe_load(open("config.yaml"))
    sys.path.append(config["SAM_REPO_DIR"])
    from medim_infer import ( # Assuming helpers are in this file
        create_gt_arr,
        data_preprocess,
        sam_model_infer,
        data_postprocess
        # random_sample_next_click # May not be needed directly if sam_model_infer uses roi_gt
    )
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import 'medim' or helper functions: {e}")
    print("Ensure 'medim' is installed and 'medim_infer_helpers.py' is accessible.")
    SAM_AVAILABLE = False

# --- Predictor Class ---

class SAMMed3DPredictor:
    """
    Minimal predictor class for SAM-Med3D using provided helper functions.
    """
    def __init__(self, checkpoint_path):
        if not SAM_AVAILABLE:
            raise ImportError("SAM-Med3D components or helpers not found.")

        print(f"Initializing SAMMed3DPredictor with checkpoint: {checkpoint_path}")
        self.model = None
        try:
            # Use medim to load the model
            self.model = medim.create_model(
                "SAM-Med3D",
                # pretrained=True, # Check if needed based on checkpoint
                checkpoint_path=checkpoint_path
            )
            # Optional: Move to GPU
            if torch.cuda.is_available():
                 self.model.to('cuda')
            print("SAMMed3DPredictor: Model loaded.")
        except Exception as e:
            print(f"ERROR loading SAM-Med3D model from {checkpoint_path}: {e}")
            raise RuntimeError(f"Failed to load SAM-Med3D model from {checkpoint_path}") from e

    def predict(self,
                image_data,        # Full 3D image (NumPy array)
                spacing_data,      # Voxel spacing (tuple/list) - NB: Helpers might expect zyx? Check data_preprocess usage.
                bbox_data=None,    # BBox list (iter 0) [{'z_min':..,}, ...]
                clicks_data=None,  # Click list (iter 1+) [{'fg':[[x,y,z],..], 'bg':[[x,y,z],..]}, ...]
                prev_pred_data=None # Full prediction from previous step (NumPy array)
               ):
        """
        Performs SAM-Med3D inference using the provided helper functions.
        Mimics the logic flow from the example script's main block.
        """
        start_time = time.time()
        print("--- SAMMed3DPredictor.predict() start (Using Helpers) ---")

        if self.model is None:
             print("   ERROR: Model is not loaded.")
             return np.zeros_like(image_data, dtype=np.uint8), time.time() - start_time

        # Initialize final prediction for THIS iteration step
        final_segmentation = np.zeros_like(image_data, dtype=np.uint8)

        # Ensure prev_pred_data is an array
        if prev_pred_data is None:
            prev_pred_data = np.zeros_like(image_data, dtype=np.uint8)

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

        # --- Iterate through classes ---
        for idx in range(num_classes_prompted):
            class_index = idx + 1 # SAM uses 1-based indexing
            print(f"--- Processing Class Index: {class_index} ---")

            # Extract previous prediction mask for this class
            cls_prev_seg_mask = (prev_pred_data == class_index).astype(np.uint8)

            # --- Copy previous prediction to output (will be overwritten if inference succeeds) ---
            final_segmentation[cls_prev_seg_mask != 0] = class_index
            # print(f"   Copied previous prediction for class {class_index}. Voxels: {np.sum(cls_prev_seg_mask)}")

            # --- Determine the single point prompt for create_gt_arr ---
            # Mimic the logic `cls_clicks[-1][0]` and `if (cls_clicks[-1][1][0] == 1):`
            # We need ONE point and its label (positive/negative) based on the LAST interaction
            prompt_point = None
            is_positive_prompt = False # Default, only proceed if we find a positive prompt

            if is_bbox_iteration:
                bbox = bbox_data[idx]
                # Calculate center point of the bbox (Ensure coordinate order: z, y, x for create_gt_arr?)
                # Example bbox format might be complex, assuming simple calculation for now
                # Verify coordinate system needed by create_gt_arr
                z_cen = (bbox['z_min'] + bbox['z_max']) / 2
                y_cen = (bbox['z_mid_y_min'] + bbox['z_mid_y_max']) / 2
                x_cen = (bbox['z_mid_x_min'] + bbox['z_mid_x_max']) / 2
                prompt_point = (z_cen, y_cen, x_cen)
                is_positive_prompt = True # Bbox is always a positive prompt
                print(f"   Using BBox center as prompt point: {prompt_point}")

            else: # Click iteration
                fg_clicks = clicks_data[idx].get('fg', [])
                bg_clicks = clicks_data[idx].get('bg', [])

                # Find the "last" click conceptually - For simplicity, prioritize last FG, then last BG
                if fg_clicks: # If there are foreground clicks in this batch
                    # The evaluation framework adds clicks one by one, so the last in the list IS the newest
                    last_fg_click = fg_clicks[-1]
                    # Ensure coordinate order (example: clicks are [x,y,z], create_gt_arr wants [z,y,x]?)
                    # Assuming create_gt_arr expects (z, y, x) based on original script's usage pattern
                    # Adjust if needed: prompt_point = (last_fg_click[2], last_fg_click[1], last_fg_click[0])
                    prompt_point = (last_fg_click[2], last_fg_click[1], last_fg_click[0]) # SWAP TO Z,Y,X
                    is_positive_prompt = True
                    print(f"   Using last FG click as prompt point: {prompt_point}")
                elif bg_clicks: # Otherwise, if only background clicks were added
                    last_bg_click = bg_clicks[-1]
                    # prompt_point = (last_bg_click[2], last_bg_click[1], last_bg_click[0]) # SWAP TO Z,Y,X
                    prompt_point = (last_bg_click[2], last_bg_click[1], last_bg_click[0])
                    is_positive_prompt = False # It's a background click
                    print(f"   Using last BG click as prompt point: {prompt_point}")
                else:
                    # No clicks provided for this class in this iteration step
                    print(f"   Skipping class {class_index}: No clicks in this step.")
                    continue # Keep copied prediction

            # --- Mimic the `if (cls_clicks[-1][1][0] == 1):` condition ---
            # Only proceed with SAM steps if the determined prompt point is positive (Foreground/BBox)
            if not is_positive_prompt:
                print(f"   Skipping SAM steps for class {class_index}: Last prompt was negative.")
                continue # Keep copied prediction

            # --- Generate the prompt mask using the chosen point ---
            try:
                # Use the helper function with the single determined point
                # Ensure prompt_point coords are valid indices (int, within bounds)
                # Make sure coordinate order matches create_gt_arr expectation
                int_prompt_point = tuple(int(max(0, min(image_data.shape[d]-1, prompt_point[d]))) for d in range(3))
                cls_gt = create_gt_arr(image_data.shape, int_prompt_point, category_index=class_index)
                print(f"   Generated cls_gt prompt mask around {int_prompt_point}.")
            except Exception as e:
                 print(f"   ERROR generating cls_gt prompt for class {class_index}: {e}")
                 continue # Keep copied prediction


            # --- Run SAM Steps using Helpers ---
            try:
                # 1. Preprocess: Pass original image, generated prompt mask, prev seg mask
                # Check if data_preprocess needs specific spacing format (e.g., zyx vs xyz)
                print(f"   Preprocessing...")
                roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(
                    image_data, cls_gt, cls_prev_seg_mask,
                    orig_spacing=spacing_data, # Pass spacing directly
                    category_index=class_index
                )
                print(f"   Preprocessing done.")

                # 2. Inference: Pass model, processed ROI data
                print(f"   Running SAM inference...")
                # sam_model_infer uses roi_label (derived from cls_gt) as roi_gt
                roi_pred = sam_model_infer(
                    self.model, roi_image,
                    roi_gt=roi_label, # Pass the processed prompt mask
                    prev_low_res_mask=roi_prev_seg # Pass the processed previous mask
                )
                print(f"   Inference done.")

                # 3. Postprocess: Pass ROI prediction and metadata
                print(f"   Postprocessing...")
                pred_ori = data_postprocess(roi_pred, meta_info, output_dir=None) # No need to save here
                print(f"   Postprocessing done.")

                # 4. Aggregate Result: Overwrite final_segmentation with new prediction
                update_mask = (pred_ori != 0)
                final_segmentation[update_mask] = class_index
                print(f"   Updated final segmentation for class {class_index}. New non-zero voxels: {np.sum(update_mask)}")

            except Exception as e:
                print(f"   ERROR during SAM steps for class {class_index}: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Keeping previous prediction for class {class_index} due to error.")
                # The copied previous prediction remains

        # --- End of class loop ---
        inference_time = time.time() - start_time
        print(f"--- SAMMed3DPredictor.predict() end. Time: {inference_time:.2f}s ---")

        return final_segmentation, inference_time