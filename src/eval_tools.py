
import argparse
import os
import shutil
import time
from collections import OrderedDict

import cc3d
import numpy as np
import pandas as pd
import wandb  # Import Wandb
import yaml
from scipy import integrate

# --- Competition Metric Functions (Copied from evaluation script) ---
from scipy.ndimage import distance_transform_edt
from src.sammed3d import SAMMed3DPredictor
from surface_distance import (
    compute_dice_coefficient,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)
from tqdm import tqdm

# Optional: Check for GPU and import CuPy/cuCIM if available
try:
    import cupy as cp
    from cucim.core.operations import morphology

    # Check if GPU is actually available
    cp.cuda.Device(0).use()
    GPU_AVAILABLE = True
    print("GPU detected. Using CuPy/cuCIM for EDT.")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"GPU not available or CuPy/cuCIM error ({e}). Using SciPy for EDT.")

# Function to compute multi-class DSC
def compute_multi_class_dsc(gt, seg):
    dsc = []
    labels = np.unique(gt)
    if len(labels) > 1:  # Check if there are foreground classes
        for i in labels[1:]:  # skip bg 0
            if np.sum(gt == i) == 0 and np.sum(seg == i) == 0:
                dsc.append(1.0)  # Both GT and Pred are empty for this class
            elif np.sum(gt == i) == 0 and np.sum(seg == i) > 0:
                dsc.append(0.0)  # GT empty, Pred not empty
            elif np.sum(gt == i) > 0 and np.sum(seg == i) == 0:
                dsc.append(0.0)  # GT not empty, Pred empty
            else:
                gt_i = gt == i
                seg_i = seg == i
                dsc.append(compute_dice_coefficient(gt_i, seg_i))
        return np.mean(dsc) if dsc else 1.0  # Return 1.0 if only background exists
    else:
        return 1.0  # Only background class present


# Function to compute multi-class NSD
def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    labels = np.unique(gt)
    if len(labels) > 1:
        for i in labels[1:]:  # skip bg 0
            if np.sum(gt == i) == 0 and np.sum(seg == i) == 0:
                nsd.append(1.0)
            elif np.sum(gt == i) == 0 and np.sum(seg == i) > 0:
                nsd.append(0.0)
            elif np.sum(gt == i) > 0 and np.sum(seg == i) == 0:
                nsd.append(0.0)
            else:
                gt_i = gt == i
                seg_i = seg == i
                try:
                    surface_distance = compute_surface_distances(
                        gt_i, seg_i, spacing_mm=spacing
                    )
                    nsd.append(
                        compute_surface_dice_at_tolerance(surface_distance, tolerance)
                    )
                except Exception as e:
                    print(
                        f"Warning: NSD calculation failed for class {i}: {e}. Appending 0.0"
                    )
                    nsd.append(
                        0.0
                    )  # Handle potential errors in surface distance calculation
        return np.mean(nsd) if nsd else 1.0
    else:
        return 1.0


# --- Placeholder for Your Model Inference ---
def run_model_inference(
    image_data, spacing_data, bbox_data=None, clicks_data=None, prev_pred_data=None
):
    """
    This is where you integrate your baseline model.

    Args:
        image_data (np.ndarray): The 3D input image.
        spacing_data (tuple): Voxel spacing.
        bbox_data (list, optional): List of bbox dictionaries [{'z_min':..,}, ...]. Defaults to None.
        clicks_data (list, optional): List of click dictionaries [{'fg': [], 'bg': []}, ...]. Defaults to None.
        prev_pred_data (np.ndarray, optional): Segmentation from the previous iteration. Defaults to None.

    Returns:
        tuple: (segmentation_mask (np.ndarray), inference_time (float))
               The segmentation mask should have integer labels matching the GT.
    """
    start_time = time.time()

    # #############################################
    # ##  YOUR BASELINE MODEL INFERENCE CODE HERE ##
    # ##  - Preprocess inputs as needed         ##
    # ##  - Run model prediction                ##
    # ##  - Postprocess output to segmentation mask ##
    # #############################################

    # Example placeholder: return zeros matching image shape
    print("--- Running Placeholder Inference ---")
    if bbox_data:
        print(f"Received BBox: {bbox_data}")
    if clicks_data:
        print(f"Received Clicks: {clicks_data}")
    if prev_pred_data is not None:
        print(f"Received Previous Prediction Shape: {prev_pred_data.shape}")

    # Simulate some processing time
    time.sleep(0.5)  # Replace with actual model runtime

    # Make sure the output is an integer mask of the same shape as the input
    predicted_segmentation = np.zeros_like(image_data, dtype=np.uint8)

    # #############################################

    inference_time = time.time() - start_time
    return predicted_segmentation, inference_time


# --- Click Generation Logic ---
def generate_clicks(pred_seg, gt_seg, current_clicks_per_class, verbose=False):
    """Generates one click (fg or bg) for each class based on the largest error."""
    new_clicks = [
        {"fg": list(c["fg"]), "bg": list(c["bg"])} for c in current_clicks_per_class
    ]  # Deep copy
    num_classes = len(current_clicks_per_class)
    gt_labels = sorted(np.unique(gt_seg)[1:])  # Get foreground labels from GT

    if len(gt_labels) != num_classes:
        print(
            f"[Warning] Mismatch between GT labels ({len(gt_labels)}) and number of click structures ({num_classes}). Using GT labels."
        )
        # Adjust click structure if needed (less robust, better to ensure consistency upstream)
        # This part might need refinement based on how classes are handled if they disappear/appear
        num_classes = len(gt_labels)
        new_clicks = [
            {"fg": [], "bg": []} for _ in gt_labels
        ]  # Re-initialize if mismatch

    for i, cls_label in enumerate(gt_labels):
        pred_cls = (pred_seg == cls_label).astype(np.uint8)
        gt_cls = (gt_seg == cls_label).astype(np.uint8)

        error_mask = (pred_cls != gt_cls).astype(np.uint8)

        if np.sum(error_mask) == 0:
            if verbose:
                print(f"Class {cls_label}: No errors found. No click added.")
            continue  # Perfect prediction for this class

        # Find connected components of the error
        # Ensure error_mask is C-contiguous
        error_mask_c = np.ascontiguousarray(error_mask)
        errors_labeled, num_components = cc3d.connected_components(
            error_mask_c, connectivity=26, return_N=True
        )

        if num_components == 0:
            if verbose:
                print(
                    f"Class {cls_label}: No error components found by cc3d (check mask sum). No click added."
                )
            continue

        # Calculate sizes (handling potential empty labels array)
        if errors_labeled.max() > 0:
            component_sizes = np.bincount(errors_labeled.flat)
            component_sizes[0] = 0  # Ignore background
            if len(component_sizes) <= 1:  # No foreground error components found
                if verbose:
                    print(
                        f"Class {cls_label}: No foreground error components found by bincount. No click added."
                    )
                continue
            largest_component_label = np.argmax(component_sizes)
            if component_sizes[largest_component_label] == 0:
                if verbose:
                    print(
                        f"Class {cls_label}: Largest error component has size 0. No click added."
                    )
                continue
        else:  # Only background label exists in errors_labeled
            if verbose:
                print(
                    f"Class {cls_label}: errors_labeled max value is 0. No click added."
                )
            continue

        largest_component_mask = errors_labeled == largest_component_label

        # --- Distance Transform to find center ---
        # Get bounding box of the largest error component to limit computation
        coords = np.argwhere(largest_component_mask)
        if coords.size == 0:
            if verbose:
                print(f"Class {cls_label}: Largest component mask is empty. Skipping.")
            continue
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0) + 1

        # Crop error mask to the bounding box
        cropped_mask = largest_component_mask[
            min_coords[0] : max_coords[0],
            min_coords[1] : max_coords[1],
            min_coords[2] : max_coords[2],
        ]

        # Compute distance transform
        if cropped_mask.sum() == 0:  # Safety check
            if verbose:
                print(f"Class {cls_label}: Cropped mask is empty. Skipping.")
            continue

        if GPU_AVAILABLE:
            try:
                # Ensure boolean or uint8 for cuCIM
                error_mask_cp = cp.asarray(cropped_mask, dtype=cp.bool_)
                edt_cp = morphology.distance_transform_edt(error_mask_cp)
                # Find center on GPU and transfer back
                center_in_crop_flat = cp.argmax(edt_cp)
                center_in_crop = cp.unravel_index(center_in_crop_flat, edt_cp.shape)
                center_in_crop = tuple(
                    int(c) for c in center_in_crop
                )  # to tuple of ints
                del edt_cp, error_mask_cp  # Free GPU memory
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as gpu_edt_error:
                print(
                    f"Warning: GPU EDT failed for class {cls_label}: {gpu_edt_error}. Falling back to CPU."
                )
                edt = distance_transform_edt(cropped_mask)
                center_in_crop = np.unravel_index(np.argmax(edt), edt.shape)
        else:
            edt = distance_transform_edt(cropped_mask)
            center_in_crop = np.unravel_index(np.argmax(edt), edt.shape)

        # Calculate center coordinates in the original image space
        # Make sure they are Python int for JSON serialization if needed later
        center = tuple(int(min_coords[d] + center_in_crop[d]) for d in range(3))

        # Check if click is inside the error mask (sanity check)
        if not largest_component_mask[center]:
            # This can happen due to edge effects or EDT nuances.
            # As a fallback, find *any* coordinate within the largest component.
            # This is less ideal than the center but ensures a click is placed.
            fallback_coords = np.argwhere(largest_component_mask)
            if fallback_coords.size > 0:
                center = tuple(
                    int(c) for c in fallback_coords[0]
                )  # Take the first found coordinate
                if verbose:
                    print(
                        f"Class {cls_label}: EDT center outside mask, using fallback coord: {center}"
                    )
            else:
                # Should not happen if largest_component_mask was not empty earlier
                if verbose:
                    print(
                        f"Class {cls_label}: Cannot find any coord in largest component mask. Skipping click."
                    )
                continue

        # Determine click type (Foreground or Background)
        # Check the GROUND TRUTH label at the click location
        if gt_cls[center] == 1:  # Error is an undersegmentation (GT is 1, Pred is 0)
            click_type = "fg"
            assert (
                pred_cls[center] == 0
            ), f"Click logic error: FG click target should be 0 in pred, but is {pred_cls[center]}"
        else:  # Error is an oversegmentation (GT is 0, Pred is 1)
            click_type = "bg"
            assert (
                pred_cls[center] == 1
            ), f"Click logic error: BG click target should be 1 in pred, but is {pred_cls[center]}"

        # Add the click
        # Ensure click coordinates are Python lists of ints
        click_coords = [int(c) for c in center]
        new_clicks[i][click_type].append(click_coords)
        if verbose:
            print(f"Class {cls_label}: Added {click_type} click at {click_coords}")

    return new_clicks

