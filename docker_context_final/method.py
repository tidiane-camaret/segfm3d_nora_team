import os
import time
import sys
import numpy as np
import torch
import contextlib

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


class nnInteractivePredictor:
    """
    Predictor class using nnInteractive
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
        bboxs=None,  # BBox list (used in iter 0) [{'z_min':..,}, ...] * num_classes
        clicks=None,  # Click tuple (used in iter 1+) containing :
        # clicks_cls : coordinates of every fg and bg click so far, {'fg':[[x,y,z],..], 'bg':[[x,y,z],..]} * num_classes
        # clicks_order : order of clicks,  ['fg', 'fg', 'bg' ...] * num_classes
        is_bbox_iteration=True,  # True if current bbox iteration, False if click iteration
        prev_pred=None,  # Full prediction from previous step (NumPy ZYX)
        num_classes_max=None,  # Optional: limit number of classes processed
        add_previous_interactions=False,  # Optional: add previous interaction to the current one
    ):
        """
        Predicts the segmentation of the input image using nnInteractive for every class described by the bbox or click list. Called at each iteration step.

        Args:
            image (numpy.ndarray): Input 3D image (NumPy ZYX).
            spacing (tuple/list): Voxel spacing (tuple/list XYZ).
            bboxs (list, optional): List of bounding boxes for prediction.
            clicks (tuple, optional): tuple of clicks for prediction. (coordinates, bg/fg order)
            is_bbox_iteration (Bool, optional),  # True if current bbox iteration, False if click iteration
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
        if bboxs is None and clicks is None:
            self.log("   Warning: No prompts provided. Returning previous prediction.")
            return prev_pred.copy(), time.time() - start_time
        elif is_bbox_iteration:
            num_classes_prompted = len(bboxs)
            self.log(f"   Mode: BBox, Classes Prompted: {num_classes_prompted}")
        else:
            num_classes_prompted = len(clicks[1])
            self.log(f"   Mode: Clicks, Classes Prompted: {num_classes_prompted}")


        # Determine number of classes to process
        num_classes = (
            num_classes_prompted
            if num_classes_max is None
            else min(num_classes_max, num_classes_prompted)
        )
        self.log(f"Processing {num_classes} classes.")

        prev_pred = np.zeros_like(image, dtype=np.uint8) if prev_pred is None else prev_pred

        # Iteration through classes
        for idx in range(num_classes):
            self.log(f"image shape: {image.shape}")
            class_id = idx + 1  # 0 is reserved for background
            self.log(f"\n--- Processing Class Index: {class_id} ---")
            self.log(f"   Previous prediction: {prev_pred.shape}")

            ### Add the previous interaction
            self.session.reset_interactions()  # Clears the target buffer and resets interactions
            self.session.add_initial_seg_interaction(prev_pred==class_id)  # Adds the previous prediction as an interaction

            if is_bbox_iteration:
                self.log("BBOX ITERATION")
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
                with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext(): # Suppress output

                    self.session.add_bbox_interaction(
                        BBOX_COORDINATES, include_interaction=True
                    )


            else:  # click iteration TODO : Now, we add all click in temporal order, adding 
                self.log("CLICK ITERATION")
                clicks_cls, clicks_order = clicks[0][idx], clicks[1][idx]

                ordered_clicks = recover_click_sequence(clicks_cls, clicks_order)
                for click_idx, oc in enumerate(ordered_clicks):
                    is_last_click = click_idx == len(ordered_clicks) - 1
                    click_type = oc["type"]
                    click_coords = oc["coords"]
                    self.log(f"   Adding {click_type} click at {click_coords}")
                    with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext():
                        if add_previous_interactions or is_last_click:
                            # Add the click to the interaction
                            self.session.add_point_interaction(
                                click_coords, 
                                include_interaction=click_type == "fg",
                                run_prediction=is_last_click, # Only run prediction on the current click
                            )
 



            # get the prediction, add it to the final segmentation map
            results = target_tensor.clone()
            final_segmentation[results != 0] = class_id

        
            
        return final_segmentation, time.time() - start_time
    

def recover_click_sequence(clicks_cls, clicks_order):
    # Initialize empty list to store clicks in order
    ordered_clicks = []
    
    # Counters to keep track of which click we're at in each category
    bg_counter = 0
    fg_counter = 0
    
    # Iterate through the clicks_order list
    for click_type in clicks_order:
        if click_type == 'bg':
            # Get the coordinates of the next background click
            click_coords = clicks_cls['bg'][bg_counter]
            bg_counter += 1
        else:  # click_type == 'fg'
            # Get the coordinates of the next foreground click
            click_coords = clicks_cls['fg'][fg_counter]
            fg_counter += 1
        
        # Add the coordinates and type to our ordered list
        ordered_clicks.append({
            'type': click_type,
            'coords': click_coords
        })
    
    return ordered_clicks