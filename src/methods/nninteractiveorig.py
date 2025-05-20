import os
import time
import sys
import numpy as np
import torch
import contextlib
import gc
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache


class nnInteractiveOrigPredictor:
    """
    Predictor class using nnInteractive "as is".
    """
    

    def __init__(self, checkpoint_path, device, do_autozoom=True, verbose=True):
        try : 
            cores = int(os.environ['SLURM_CPUS_ON_NODE'])
        except KeyError:
            print("No SLURM environment variable found for cpu nb. Setting to os.cpu_count()")
            cores = int(os.cpu_count())
        cores_used = max(4, cores - 4)
        print(f"{cores} CPU cores avaiables")
        print(f"Using {cores_used} CPU cores for inference.")
        self.session = nnInteractiveInferenceSession(
            device=device,  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=verbose,
            torch_n_threads=cores_used,  # Use available CPU cores
            do_autozoom=do_autozoom,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )

        self.session.initialize_from_trained_model_folder(checkpoint_path)
        self.verbose = verbose
        self.session.verbose = verbose
        self.device = device
            
    
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
        add_previous_interactions=True,  # Optional: add previous interaction to the current one
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
        clicks, clicks_order = clicks # nn code logic from  https://github.com/MIC-DKFZ/nnInteractive/blob/9192ecf75f2302c4efc7d7657a083c8e967b2ad8/nnInteractive/inference/cvpr2025_challenge_baseline/predict.py
        with torch.autocast(device_type=str(self.device)):
            
            self.session.set_image(image[None].astype(np.float32))
            target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
            self.session.set_target_buffer(target_buffer)
            result = torch.zeros(image.shape, dtype=torch.uint8)
            
            num_objects = len(bboxs) if bboxs is not None else len(clicks)
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

            # Iteration through classes
            for oid in range(1, num_objects + 1):
                self.log(f"image shape: {image.shape}")
                self.log(f"\n--- Processing Class Index: {oid} ---")
                if prev_pred is not None:
                    self.session.add_initial_seg_interaction((prev_pred == oid).astype(np.uint8), run_prediction=False)
                else:
                    self.session.reset_interactions()

                if (bboxs is not None) and is_bbox_iteration:
                    bbox_here = bboxs[oid - 1]
                    bbox_here = [
                        [bbox_here['z_min'], bbox_here['z_max'] + 1],
                        [bbox_here['z_mid_y_min'], bbox_here['z_mid_y_max'] + 1],
                        [bbox_here['z_mid_x_min'], bbox_here['z_mid_x_max'] + 1]
                        ]
                
                    with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext(): # Suppress output
                        self.log(f"   BBox coordinates: {bbox_here}")
                        self.session.add_bbox_interaction(bbox_here, include_interaction=True, run_prediction=False)


                else:  # click iteration TODO : Now, we add all click in temporal order, adding 
                    self.log("CLICK ITERATION")
                    if clicks is not None:

                        clicks_here = clicks[oid - 1]
                        clicks_order_here = clicks_order[oid - 1]
                        fg_ptr = bg_ptr = 0
                        
                        for i_click, kind in enumerate(clicks_order_here):
                            if kind == 'fg':
                                click = clicks_here['fg'][fg_ptr]
                                fg_ptr += 1
                            else:
                                click = clicks_here['bg'][bg_ptr]
                                bg_ptr += 1

                            print(f"Class {oid}: {kind} click at {click}")
                            is_last_click = i_click == (len(clicks_order_here) - 1)
                            with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext():
                                if add_previous_interactions or is_last_click:
                                    self.session.add_point_interaction(click, include_interaction=kind == 'fg', run_prediction=False)

                    #clicks_cls, clicks_order = clicks[0][oid - 1], clicks[1][oid - 1]

                    # ordered_clicks = recover_click_sequence(clicks_cls, clicks_order)
                    # for click_idx, oc in enumerate(ordered_clicks):
                    #     is_last_click = click_idx == len(ordered_clicks) - 1
                    #     click_type = oc["type"]
                    #     click_coords = oc["coords"]
                    #     self.log(f"   Adding {click_type} click at {click_coords}")
                    #             # Add the click to the interaction
                    #             self.session.add_point_interaction(
                    #                 click_coords, 
                    #                 include_interaction=click_type == "fg",
                    #                 run_prediction=False, 
                    #             )
    

                self.session.verbose = self.verbose
                self.session.new_interaction_centers = [self.session.new_interaction_centers[-1]]
                self.session.new_interaction_zoom_out_factors = [self.session.new_interaction_zoom_out_factors[-1]]
                with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext():
                    self.session._predict()
                forward_pass_count = self.session.forward_pass_count
                # get the prediction, add it to the final segmentation map

                
                result[self.session.target_buffer > 0] = oid
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    empty_cache(torch.device('cuda', 0))
                gc.collect()

            self.session._reset_session()

            prediction_metrics = {"infer_time": time.time() - start_time,
                    "forward_pass_count": forward_pass_count}
                
            return result.cpu().numpy(), prediction_metrics
    