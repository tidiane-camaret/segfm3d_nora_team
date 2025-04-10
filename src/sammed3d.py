# predictor class based on SAM-Med3D

import os  # For saving debug files
import sys
import time

from click import prompt
import numpy as np
from sympy import im
import torch
import torch.nn.functional as F
from onnx import save
from src.viz_tools import save_volume_viz

try:
    import medim  # model loading
    import yaml

    config_path = "config.yaml"
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

    from medim_infer import (  # processing helpers
        data_postprocess,
        data_preprocess,
        random_sample_next_click,
        create_gt_arr,
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

    def __init__(self, checkpoint_path):
        """
        Initializes the predictor, loads the model

        Args:
            checkpoint_path (str): Path to the model checkpoint.
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM-Med3D components or helpers not found.")

        print(f"Initializing SAMMed3DPredictor with checkpoint: {checkpoint_path}")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")

        try:
            # Use medim to load the model
            self.model = medim.create_model(
                "SAM-Med3D",
                pretrained=True, # Check if needed based on checkpoint
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
        image,  # Full 3D image (NumPy ZYX)
        spacing,  # Voxel spacing (tuple/list XYZ)
        bboxs=None,  # BBox list (iter 0) [{'z_min':..,}, ...]
        clicks=None,  # Click list (iter 1+) [{'fg':[[x,y,z],..], 'bg':[[x,y,z],..]}, ...]
        prev_pred=None,  # Full prediction from previous step (NumPy ZYX)
        num_classes_max=None,  # Optional: limit number of classes processed
    ):
        """
        Performs SAM-Med3D inference using the provided helper functions.
        Includes detailed debugging logs and saves intermediate arrays.
        """
        start_time = time.time()
        # Create a unique prefix for debug files for this specific call
        if self.model is None:
            print("   ERROR: Model is not loaded.")
            return np.zeros_like(image, dtype=np.uint8), time.time() - start_time

        # Initialize final prediction array
        final_segmentation = np.zeros_like(image, dtype=np.uint8)

        # Ensure prev_pred_data is an array
        if prev_pred is None:
            print("   prev_pred is None, initializing as zeros.")
            prev_pred = np.zeros_like(image, dtype=np.uint8)
        else:
            print(
                f"   Received prev_pred shape: {prev_pred.shape}, dtype: {prev_pred.dtype}"
            )
            # Save initial prev_pred for this iteration

        # Determine number of classes and iteration type (bbox or clicks)
        num_classes_prompted = 0
        is_bbox_iteration = False
        if bboxs is not None:
            num_classes_prompted = len(bboxs)
            is_bbox_iteration = True
            print(f"   Mode: BBox, Classes Prompted: {num_classes_prompted}")
        elif clicks is not None:
            num_classes_prompted = len(clicks)
            print(f"   Mode: Clicks, Classes Prompted: {num_classes_prompted}")
        else:
            print("   Warning: No prompts provided. Returning previous prediction.")
            return prev_pred.copy(), time.time() - start_time

        # Determine number of classes to process
        num_classes = (
            num_classes_prompted
            if num_classes_max is None
            else min(num_classes_max, num_classes_prompted)
        )
        print(f"   Processing up to {num_classes} classes.")

        # --- Iterate through classes ---
        for idx in range(num_classes):
            class_id = idx + 1  # 0 is reserved for background
            print(f"\n--- Processing Class Index: {class_id} ---")

            # --- Verify Inputs for this Class ---
            cls_prev_seg = (prev_pred == class_id).astype(np.uint8)

            # --- Copy previous prediction to output (will be overwritten if inference succeeds) ---
            final_segmentation[cls_prev_seg != 0] = class_id

            print(
                f"   Copied previous prediction area for class {class_id} to initial output."
            )

            if is_bbox_iteration:
                # create a bbox mask for the current class, used later to create the roi
                bbox = bboxs[idx]
                img_bbox = np.zeros_like(image)
                img_bbox[
                    bbox["z_min"] : bbox["z_max"],
                    bbox["z_mid_y_min"] : bbox["z_mid_y_max"],
                    bbox["z_mid_x_min"] : bbox["z_mid_x_max"],
                ] = class_id
                cls_gt = img_bbox  # TODO : change this ambiguous name
                
            else:  # Click iteration
                fg_clicks = clicks[idx].get(
                    "fg", []
                )  # Expected format: list of [z, y, x]
                bg_clicks = clicks[idx].get(
                    "bg", []
                )  # Expected format: list of [z, y, x]
                print(f"   Foreground clicks: {fg_clicks}")
                print(f"   Background clicks: {bg_clicks}")
                if fg_clicks:
                    prompt_point = fg_clicks[-1]
                    is_positive_prompt = True
                    print(
                        f"   FG click (ZYX): {prompt_point}"
                    )

                    cls_gt = create_gt_arr(image.shape, prompt_point, category_index=class_id)
                    # just a bbox around the center point (size 5)

                elif bg_clicks:
                    """
                    prompt_point = bg_clicks[-1]
                    """
                    prompt_point = (image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2)
                    is_positive_prompt = False
                    print(
                        f"   BG click. We just use the center for now: (ZYX){prompt_point}"
                    )
                    # TODO : How to deal with background clicks?
                    # the model deals with bbox and positive clicks by 
                    # creating a mask around them.
                    # But what about background clicks?
                    # For now we create a mask around the center of the image
                    # This is really bad. Lets find something better.
                    cls_gt = create_gt_arr(image.shape, prompt_point, category_index=class_id)

                else:
                    print(f"   Skipping class {class_id}: No clicks in this step.")
                    continue
            print(f"unique values in cls_gt: {np.unique(cls_gt, return_index=True)}")
            save_volume_viz(
                image, slice_indices=list(range(34, 51, 4)), save_path="debug_image.png"
            )
            save_volume_viz(
                cls_gt,
                slice_indices=list(range(34, 51, 4)),
                save_path="debug_cls_gt.png",
            )

            # 1. Preprocess
            print(f"   Preprocessing...")
            print(f"     Input image_data shape: {image.shape}, dtype: {image.dtype}")
            print(
                f"     Input cls_gt shape: {cls_gt.shape}, dtype: {cls_gt.dtype}, sum: {np.sum(cls_gt)}"
            )
            print(
                f"     Input cls_prev_seg_mask shape: {cls_prev_seg.shape}, dtype: {cls_prev_seg.dtype}"
            )

            spacing = [
                spacing[2],
                spacing[0],
                spacing[1],
            ]  # order used by data_preprocess

            roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(
                image,
                cls_gt,
                cls_prev_seg,
                orig_spacing=spacing,
                category_index=class_id,
            )

            print(f"     ROI image shape: {roi_image.shape}, dtype: {roi_image.dtype}")
            print(f"     ROI label shape: {roi_label.shape}, dtype: {roi_label.dtype}")
            print(
                f"     ROI prev_seg shape: {roi_prev_seg.shape}, dtype: {roi_prev_seg.dtype}"
            )
            print(
                f"     Meta info: {meta_info}, spacing: {spacing}, class_id: {class_id}"
            )
            """
            save_volume_viz(
                roi_image[0][0].cpu().numpy(),
                slice_indices=None,
                save_path="debug_roi_image.png",
            )
            save_volume_viz(
                roi_label[0][0].cpu().numpy(),
                slice_indices=None,
                save_path="debug_roi_label.png",
            )
            
            save_volume_viz(
                roi_prev_seg[0][0].cpu().numpy(),
                slice_indices=None,
                save_path="debug_roi_prev_seg.png",
            )
            """            
            # 2. Model predictions
            print(f"   Running SAM-Med3D inference...")

            # 2.1. Get the image embeddings
            image_embeddings = self.model.image_encoder(roi_image.to(self.device))
            print(f"image_embeddings shape: {image_embeddings.shape}")
            

            # 2.2. Get the prompt embeddings
            # This is ugly, we can do better
            # Click info is not passed to the model,
            # A click is generated within the ROI
            # TODO: see what we can do with the actual click info
            # Bbox info is not passed to the model either
            # (but used previously to create the ROI, see data_preprocess)

            # initialize empty tensors for points_coords and points_labels
            points_coords, points_labels = torch.zeros(1, 0, 3).to(
                self.device
            ), torch.zeros(1, 0).to(self.device)

            # by default, add a positive point at the center of the image
            new_points_co, new_points_la = torch.Tensor([[[64, 64, 64]]]).to(
                self.device
            ), torch.Tensor([[1]]).to(torch.int64)
            
            # if we have a previous prediction,
            # we can use it to create smarter points : 

            # ensuring existence and correct shape of roi_prev_seg
            if roi_label is not None:
                roi_prev_seg = (
                    roi_prev_seg
                    if (roi_prev_seg is not None)
                    else torch.zeros(
                        1,
                        1,
                        roi_image.shape[2] // 4,
                        roi_image.shape[3] // 4,
                        roi_image.shape[4] // 4,
                    )
                )
                # 
                roi_prev_seg = F.interpolate(
                    roi_prev_seg,
                    size=(
                        roi_image.shape[2] // 4,
                        roi_image.shape[3] // 4,
                        roi_image.shape[4] // 4,
                    ),
                    mode="nearest",
                ).to(torch.float32)

                print(np.unique(roi_label.cpu().numpy()))

                # finds all wrong pixels, randomly picks one,
                # returns click information formatted for model input.
                # TODO : As of its written, its simply chosing a random pixel
                # from roi_label and labeling it as a positive click
                prompt_generator = random_sample_next_click
                new_points_co, new_points_la = prompt_generator(
                    torch.zeros_like(roi_image)[0, 0], roi_label[0, 0]
                )
                new_points_co, new_points_la = new_points_co.to(
                    self.device
                ), new_points_la.to(self.device)
            points_coords = torch.cat([points_coords, new_points_co], dim=1)
            points_labels = torch.cat([points_labels, new_points_la], dim=1)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=[points_coords, points_labels],
                boxes=None,  # bbox is not handeled by the model TODO check anyway
                masks = roi_prev_seg.to(self.device), # TODO providing no mask kills performance
            )

            # 2.3. Get the mask predictions
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
                sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
            )



            print(f"low_res_masks shape: {low_res_masks.shape}")
            """
            save_volume_viz(
                low_res_masks[0][0].cpu().detach().numpy(),
                slice_indices=None,
                save_path="debug_low_res_masks.png",
            )
            """
            # 3. Postprocess

            print(f"   Postprocessing...")
            prev_mask = F.interpolate(
                low_res_masks,
                size=roi_image.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

            print(f"prev_mask shape: {prev_mask.shape}")
            """
            save_volume_viz(
                prev_mask[0][0].cpu().detach().numpy(),
                slice_indices=list(range(34, 51, 4)),
                save_path="debug_prev_mask.png",
            )
            """
            medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, 64, 64, 64)
            medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
            medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)
            print(f"medsam_seg_prob shape: {medsam_seg_prob.shape}")
            """
            save_volume_viz(
                medsam_seg_prob,
                slice_indices=list(range(34, 51, 4)),
                save_path="debug_medsam_seg_prob.png",
            )
            """
            pred_ori = data_postprocess(medsam_seg_mask, meta_info)
            final_segmentation[pred_ori != 0] = class_id

            """
            save_volume_viz(
                pred_ori,
                slice_indices=list(range(34, 51, 4)),
                save_path="debug_pred_ori.png",
            )
            """
        inference_time = time.time() - start_time
        print(f"--- SAMMed3DPredictor.predict() end. Time: {inference_time:.2f}s ---")

        return final_segmentation, inference_time
