import os
import time

import numpy as np
import torch

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


class nnInteractivePredictor:
    """
    Predictor class using nnInteractive "as is".
    """

    def __init__(self, checkpoint_path, device, do_autozoom=True, verbose=True):
        self.session = nnInteractiveInferenceSession(
            device=device,  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=verbose,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=do_autozoom,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )

        self.session.initialize_from_trained_model_folder(checkpoint_path)
        self.verbose = verbose
    
    def log(self, message):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

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
        Predicts the segmentation of the input image using nnInteractive for every class described by the bbox or click list

        Args:
            image (numpy.ndarray): Input 3D image (NumPy ZYX).
            spacing (tuple/list): Voxel spacing (tuple/list XYZ).
            bboxs (list, optional): List of bounding boxes for prediction.
            clicks (list, optional): List of clicks for prediction.
            prev_pred (numpy.ndarray, optional): Previous prediction.
            num_classes_max (int, optional): Limit number of classes processed.

        Returns:
            final_segmentation (numpy.ndarray): Final segmentation result.
            inference_time (float): Time taken for inference.
        """
        start_time = time.time()
        self.session.set_image(image[None])

        # Initialize final prediction array (all classes)
        final_segmentation = np.zeros_like(image, dtype=np.uint8)
        target_tensor = torch.zeros(image.shape, dtype=torch.uint8)
        self.session.set_target_buffer(target_tensor)

        # Determine if bbox or click mode, and set number of classes to predict
        is_bbox_iteration = False
        if bboxs is not None:
            num_classes_prompted = len(bboxs)
            is_bbox_iteration = True
            self.log(f"   Mode: BBox, Classes Prompted: {num_classes_prompted}")
        elif clicks is not None:
            num_classes_prompted = len(clicks)
            self.log(f"   Mode: Clicks, Classes Prompted: {num_classes_prompted}")
        else:
            self.log("   Warning: No prompts provided. Returning previous prediction.")
            return prev_pred.copy(), time.time() - start_time

        # Determine number of classes to process
        num_classes = (
            num_classes_prompted
            if num_classes_max is None
            else min(num_classes_max, num_classes_prompted)
        )
        self.log(f"Processing {num_classes} classes.")

        # Iteration through classes
        for idx in range(num_classes):
            self.log(f"image shape: {image.shape}")


            self.session.reset_interactions()  # Clears the target buffer and resets interactions

            class_id = idx + 1  # 0 is reserved for background
            self.log(f"\n--- Processing Class Index: {class_id} ---")

            if is_bbox_iteration:
                bbox = bboxs[idx]
                z_slice = (
                    bbox["z_min"] + bbox["z_max"]
                ) // 2  # TODO : For now, we take the middle slice in the z axis, see if other axis are better. Also, see if changing the overall axis order is better (nnInteractive is designed for XYZ order)
                BBOX_COORDINATES = [
                    [z_slice, z_slice + 1],
                    [bbox["z_mid_y_min"], bbox["z_mid_y_max"]],
                    [bbox["z_mid_x_min"], bbox["z_mid_x_max"]],
                ]
                self.log(f"   BBox coordinates: {BBOX_COORDINATES}")
                self.session.add_bbox_interaction(
                    BBOX_COORDINATES, include_interaction=True
                )


            else:  # click iteration TODO : The eval script passes all of the past clicks. Right now, we simply extract the last one.
                fg_clicks = clicks[idx].get(
                    "fg", []
                )  # Expected format: list of [z, y, x]
                bg_clicks = clicks[idx].get(
                    "bg", []
                )  # Expected format: list of [z, y, x]
                self.log(f"   Foreground clicks: {fg_clicks}")
                self.log(f"   Background clicks: {bg_clicks}")

                if fg_clicks:
                    click_coords = fg_clicks[-1]
                    self.log(f" using the last foreground click: {click_coords}")

                    self.session.add_point_interaction(
                        click_coords, include_interaction=True
                    )
                    
                else:
                    self.log(f"   Skipping class {class_id}: No positive clicks") # TODO : handle negetive clicks

            # get the prediction, add it to the final segmentation map
            results = target_tensor.clone()
            final_segmentation[results != 0] = class_id

        return final_segmentation, time.time() - start_time