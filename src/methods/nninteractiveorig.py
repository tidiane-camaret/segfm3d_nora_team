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
        with torch.autocast(device_type=str(self.device)):
            self.session.set_image(image[None].astype(np.float32))
            target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
            self.session.set_target_buffer(target_buffer)
            result = torch.zeros(image.shape, dtype=torch.uint8)
            
            num_objects = len(bboxs) if bboxs is not None else len(clicks[1])
            
            # Validate bbox and clicks lengths if both are provided
            if bboxs is not None and clicks is not None:
                assert len(bboxs) == len(clicks[1]), ('Both bboxs and clicks lists are provided but with different length '
                                                    'suggesting different number of objects. This is not supported by this script '
                                                    'and it was not communicated by the organizing team that such cases exist '
                                                    'or how they are supposed to be handled.')

            if bboxs is None and clicks is None:
                self.log("   Warning: No prompts provided. Returning previous prediction.")
                return prev_pred.copy(), time.time() - start_time

            # Iteration through classes
            for oid in range(1, num_objects + 1):
                self.log(f"image shape: {image.shape}")
                self.log(f"\n--- Processing Class Index: {oid} ---")
                
                # Reset or add previous prediction
                if prev_pred is not None:
                    self.session.add_initial_seg_interaction((prev_pred == oid).astype(np.uint8), run_prediction=False)
                else:
                    self.session.reset_interactions()

                # Process bbox if in bbox iteration
                if (bboxs is not None) and is_bbox_iteration:
                    bbox_here = bboxs[oid - 1]
                    bbox_here = [
                        [bbox_here['z_min'], bbox_here['z_max'] + 1],
                        [bbox_here['z_mid_y_min'], bbox_here['z_mid_y_max'] + 1],
                        [bbox_here['z_mid_x_min'], bbox_here['z_mid_x_max'] + 1]
                    ]
                    with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext():
                        self.log(f"   BBox coordinates: {bbox_here}")
                        self.session.add_bbox_interaction(bbox_here, include_interaction=True, run_prediction=False)

                # Process clicks if in click iteration
                elif clicks is not None:
                    self.log("CLICK ITERATION")
                    clicks_here = clicks[0][oid - 1]
                    clicks_order_here = clicks[1][oid - 1]
                    fg_ptr = bg_ptr = 0
                    
                    for kind in clicks_order_here:
                        if kind == 'fg':
                            click = clicks_here['fg'][fg_ptr]
                            fg_ptr += 1
                        else:
                            click = clicks_here['bg'][bg_ptr]
                            bg_ptr += 1

                        self.log(f"Class {oid}: {kind} click at {click}")
                        with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext():
                            self.session.add_point_interaction(click, include_interaction=kind == 'fg', run_prediction=False)

                # Run prediction
                self.session.verbose = self.verbose
                self.session.new_interaction_centers = [self.session.new_interaction_centers[-1]]
                self.session.new_interaction_zoom_out_factors = [self.session.new_interaction_zoom_out_factors[-1]]
                with contextlib.redirect_stdout(open(os.devnull, 'w')) if not self.verbose else contextlib.nullcontext():
                    self.session._predict()
                
                # Update result
                result[self.session.target_buffer > 0] = oid

            # Cleanup
            self.session._reset_session()
            if self.device == "cuda":
                empty_cache(torch.device('cuda', 0))

            prediction_metrics = {
                "infer_time": time.time() - start_time,
                "forward_pass_count": self.session.forward_pass_count
            }
                
            return result.cpu().numpy(), prediction_metrics