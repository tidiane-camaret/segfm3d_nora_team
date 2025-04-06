"""
Main evaluation script,
Adapted from https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/CVPR25_iter_eval.py
to run locally without docker
TODO : Make sure that the metrics are in line with the competition
"""

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
from baselines.sammed3d_class import SAMMed3DPredictor
from scipy import integrate

# --- Competition Metric Functions (Copied from evaluation script) ---
from scipy.ndimage import distance_transform_edt
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

config = yaml.safe_load(open("config.yaml"))
predictor = SAMMed3DPredictor(checkpoint_path=config["SAM_CKPT_PATH"])

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


# --- Main Evaluation Function ---
def evaluate(
    img_dir,
    gt_dir,
    output_dir,
    num_clicks=5,
    num_cases=10,
    use_wandb=True,
    wandb_project="segfm3d_nora_team",
    verbose=False,
):

    if use_wandb:
        # Initialize WandB
        wandb.init(
            project=wandb_project,
            config={
                "img_dir": img_dir,
                "gt_dir": gt_dir,
                "num_clicks": num_clicks,
                "evaluation_mode": "local_script",
                "gpu_available": GPU_AVAILABLE,
            },
        )
        # Define metrics for WandB summary
        wandb.define_metric("Case/DSC_AUC", summary="mean")
        wandb.define_metric("Case/NSD_AUC", summary="mean")
        wandb.define_metric("Case/DSC_Final", summary="mean")
        wandb.define_metric("Case/NSD_Final", summary="mean")
        wandb.define_metric("Case/TotalRunningTime", summary="mean")

    os.makedirs(output_dir, exist_ok=True)
    cases = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])
    if num_cases > 0:
        cases = cases[:num_cases]
    if len(cases) == 0:
        print("No cases found in the input directory.")
        return
    print(f"Cases to evaluate: {cases}")

    all_case_metrics = []

    for case_filename in tqdm(cases, desc="Evaluating Cases"):
        case_name = os.path.splitext(case_filename)[0]
        input_filepath = os.path.join(img_dir, case_filename)
        gt_filepath = os.path.join(gt_dir, case_filename)

        if not os.path.exists(gt_filepath):
            print(f"Warning: Ground truth file not found for {case_name}. Skipping.")
            continue

        try:
            # Load data
            data = np.load(input_filepath)
            gt_data = np.load(gt_filepath)

            image = data["imgs"]
            spacing = data["spacing"]
            gts = gt_data["gts"]
            initial_bbox = data.get("bbox", None)  # Use .get for optional keys

            num_classes = len(np.unique(gts)[1:])  # Number of foreground classes
            if num_classes == 0:
                print(
                    f"Warning: Case {case_name} has no foreground classes in GT. Skipping."
                )
                continue

            # --- Iteration Loop ---
            current_clicks = [{"fg": [], "bg": []} for _ in range(num_classes)]
            prev_prediction = None
            case_metrics = {"case": case_name, "iterations": []}
            all_segs_for_case = []
            total_inference_time = 0

            for it in range(num_clicks + 1):
                iter_start_time = time.time()
                iteration_log = {}

                # Prepare inputs for the model
                bbox_input = initial_bbox if it == 0 else None
                clicks_input = current_clicks if it > 0 else None

                if it == 0 and bbox_input is None:
                    print(
                        f"Case {case_name}, Iteration {it}: No bounding box provided. Running inference without bbox."
                    )
                    # Handle this in your run_model_inference if necessary
                    # Often, the first pred might be zero or based on image stats if no bbox

                print(f"\n--- Case: {case_name}, Iteration: {it} ---")
                if bbox_input:
                    print("Input: BBox")
                if clicks_input:
                    print(f"Input: Clicks - {clicks_input}")

                # Run model inference
                current_segmentation, infer_time = predictor.predict(
                    image_data=image,
                    spacing_data=spacing,
                    bbox_data=bbox_input,
                    clicks_data=clicks_input,
                    prev_pred_data=prev_prediction,  # Pass previous prediction
                )
                total_inference_time += infer_time
                all_segs_for_case.append(
                    current_segmentation.astype(np.uint8)
                )  # Store prediction

                # Calculate metrics for this iteration
                dsc = compute_multi_class_dsc(gts, current_segmentation)
                nsd = 0.0
                # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
                if dsc > 0.2:
                    try:
                        nsd = compute_multi_class_nsd(
                            gts, current_segmentation, spacing
                        )
                    except Exception as nsd_err:
                        print(
                            f"Error calculating NSD for {case_name} iter {it}: {nsd_err}"
                        )
                        nsd = 0.0  # Assign 0 if calculation fails
                else:
                    nsd = 0.0

                print(
                    f"Iter {it} - DSC: {dsc:.4f}, NSD: {nsd:.4f}, Time: {infer_time:.2f}s"
                )

                # Store iteration metrics
                iteration_log = {
                    "iteration": it,
                    "DSC": dsc,
                    "NSD": nsd,
                    "Time": infer_time,
                }
                case_metrics["iterations"].append(iteration_log)

                # Log per-iteration metrics to WandB (prefix with Case/Iter)
                if use_wandb:
                    wandb.log(
                        {
                            f"Iteration/DSC": dsc,
                            f"Iteration/NSD": nsd,
                            f"Iteration/Time": infer_time,
                            "case_step": f"{case_name}_iter_{it}",  # Custom step for viewing per iteration
                            "global_step": len(all_case_metrics) * (num_clicks + 1)
                            + it,  # Overall step counter
                        }
                    )

                # --- Generate clicks for the *next* iteration (if not the last one) ---
                if it < num_clicks:
                    print("Generating clicks for next iteration...")
                    # Make sure prev_prediction is set correctly for click generation
                    # It should be the prediction *from this iteration*
                    prev_prediction_for_clickgen = current_segmentation
                    current_clicks = generate_clicks(
                        pred_seg=prev_prediction_for_clickgen,
                        gt_seg=gts,
                        current_clicks_per_class=current_clicks,  # Pass the *state* before this iter's clicks
                        verbose=verbose,
                    )
                    # Update the overall previous prediction state for the *next* model run
                    prev_prediction = current_segmentation
                else:
                    # Save final prediction and intermediate steps
                    final_pred_path = os.path.join(output_dir, f"{case_name}_pred.npz")
                    np.savez_compressed(
                        final_pred_path,
                        segs=current_segmentation.astype(np.uint8),
                        all_segs=np.stack(all_segs_for_case, axis=0).astype(
                            np.uint8
                        ),  # Stack along a new axis
                    )
                    print(
                        f"Saved final prediction and intermediates to {final_pred_path}"
                    )

            # --- Calculate Case Summary Metrics (AUC, Final) ---
            dsc_scores = [m["DSC"] for m in case_metrics["iterations"]]
            nsd_scores = [m["NSD"] for m in case_metrics["iterations"]]
            times = [m["Time"] for m in case_metrics["iterations"]]

            # AUC calculated over the CLICK iterations (index 1 to N)
            # Need at least 2 points for trapezoid rule
            dsc_auc = 0.0
            nsd_auc = 0.0
            if num_clicks >= 1:  # Need at least 1 click iteration (iteration 1 onwards)
                click_dscs = dsc_scores[1:]  # Scores from iterations 1 to num_clicks
                click_nsds = nsd_scores[1:]
                if len(click_dscs) >= 2:  # Need at least 2 click points for AUC
                    dsc_auc = integrate.cumulative_trapezoid(
                        np.array(click_dscs), dx=1
                    )[-1]
                    nsd_auc = integrate.cumulative_trapezoid(
                        np.array(click_nsds), dx=1
                    )[-1]
                elif (
                    len(click_dscs) == 1
                ):  # If only 1 click iter, AUC is not well-defined, maybe report first click score?
                    # Reporting 0 AUC for simplicity if only 1 click point.
                    print(
                        f"Case {case_name}: Only one click iteration, AUC calculated as 0."
                    )
                    dsc_auc = 0.0
                    nsd_auc = 0.0

            dsc_final = dsc_scores[-1]  # Score after the last click
            nsd_final = nsd_scores[-1]

            case_summary = {
                "CaseName": case_name,
                "DSC_AUC": dsc_auc,
                "NSD_AUC": nsd_auc,
                "DSC_Final": dsc_final,
                "NSD_Final": nsd_final,
                "TotalRunningTime": total_inference_time,
                "AvgRunningTime": total_inference_time / (num_clicks + 1),
            }
            all_case_metrics.append(case_summary)
            print(f"Case Summary: {case_summary}")

            # Log case summary metrics to WandB (prefix with Case/)
            if use_wandb:
                wandb.log(
                    {
                        "Case/DSC_AUC": dsc_auc,
                        "Case/NSD_AUC": nsd_auc,
                        "Case/DSC_Final": dsc_final,
                        "Case/NSD_Final": nsd_final,
                        "Case/TotalRunningTime": total_inference_time,
                        "Case/AvgRunningTime": case_summary["AvgRunningTime"],
                        "case_name": case_name,  # Log case name for grouping
                    }
                )

        except Exception as e:
            print(
                f"!!!!!!!!!!!!!! ERROR processing case {case_name}: {e} !!!!!!!!!!!!!!"
            )
            import traceback

            traceback.print_exc()
            # Log error to wandb if possible
            if use_wandb:
                wandb.log({"errors": 1, "error_case": case_name})

    # --- Save overall metrics ---
    if all_case_metrics:
        metrics_df = pd.DataFrame(all_case_metrics)
        csv_path = os.path.join(output_dir, "summary_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"\nSaved summary metrics to {csv_path}")
        print("\nOverall Average Metrics:")
        print(
            metrics_df[
                ["DSC_AUC", "NSD_AUC", "DSC_Final", "NSD_Final", "TotalRunningTime"]
            ].mean()
        )

        # Log overall averages to WandB summary
        if use_wandb:
            wandb.summary["Avg_DSC_AUC"] = metrics_df["DSC_AUC"].mean()
            wandb.summary["Avg_NSD_AUC"] = metrics_df["NSD_AUC"].mean()
            wandb.summary["Avg_DSC_Final"] = metrics_df["DSC_Final"].mean()
            wandb.summary["Avg_NSD_Final"] = metrics_df["NSD_Final"].mean()
            wandb.summary["Avg_TotalRunningTime"] = metrics_df[
                "TotalRunningTime"
            ].mean()
            wandb.summary["Total_Cases_Evaluated"] = len(metrics_df)
            wandb.summary["Total_Cases_Input"] = len(cases)

    if use_wandb:
        wandb.finish()


# --- Argparse and Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Local Iterative Segmentation Evaluation with WandB"
    )
    parser.add_argument(
        "-o",
        "--save_path",
        default="./results",
        type=str,
        help="Local segmentation and metrics output path",
    )
    parser.add_argument(
        "-n",
        "--num_cases",
        default=10,
        type=int,
        help="Number of cases to evaluate",
    )
    parser.add_argument(
        "-c",
        "--num_clicks",
        default=5,
        type=int,
        help="Number of click refinement iterations",
    )

    parser.add_argument(
        "--wandb_project",
        default="segfm3d_nora_team",
        type=str,
        help="WandB project name",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output for clicks etc."
    )

    args = parser.parse_args()

    
    args.val_imgs_path = os.path.join(config["VAL_DIR"], "3D_val_npz")
    args.validation_gts_path = os.path.join(
        config["VAL_DIR"], "3D_val_gt_interactive_seg"
    )

    evaluate(
        img_dir=args.val_imgs_path,
        gt_dir=args.validation_gts_path,
        output_dir=args.save_path,
        num_clicks=args.num_clicks,
        num_cases=args.num_cases,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        verbose=args.verbose,
    )
