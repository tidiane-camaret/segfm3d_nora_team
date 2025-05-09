"""
Main evaluation script,
Adapted from https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/CVPR25_iter_eval.py
to run locally without docker
TODO : Make sure that the metrics are in line with the competition
"""

import argparse
from ast import parse
from calendar import c
import os
import shutil
import time
from collections import OrderedDict

import cc3d
import numpy as np
import pandas as pd
import torch
import wandb  # Import Wandb
import yaml
from scipy import integrate

# --- Competition Metric Functions (Copied from evaluation script) ---
from scipy.ndimage import distance_transform_edt
from segfm3d_nora_team.experiments.eval_tools_old import (
    compute_multi_class_dsc,
    compute_multi_class_nsd,
    generate_clicks,
)
from src.viz_tools import plot_middle_slice, center_of_mass


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


# --- Main Evaluation Function ---
def evaluate(
    method,
    img_dir,
    gt_dir,
    output_dir,
    num_clicks=5,
    num_cases=10,
    num_classes_max=None,
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
                "method": method,
            },
        )
        # Define metrics for WandB summary
        wandb.define_metric("Case/DSC_AUC", summary="mean")
        wandb.define_metric("Case/NSD_AUC", summary="mean")
        wandb.define_metric("Case/DSC_Final", summary="mean")
        wandb.define_metric("Case/NSD_Final", summary="mean")
        wandb.define_metric("Case/TotalRunningTime", summary="mean")

    if method == "sammed3d":
        
        from src.sammed3d import SAMMed3DPredictor
        predictor = SAMMed3DPredictor(checkpoint_path=config["SAM_CKPT_PATH"])
    elif method == "nnint":
        from src.nninteractive import nnInteractivePredictor
        predictor = nnInteractivePredictor(
            checkpoint_path=os.path.join(config["NNINT_CKPT_DIR"], "nnInteractive_v1.0"),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    elif method == "nnintcore":
        from src.nninteractive import nnInteractiveCorePredictor
        predictor = nnInteractiveCorePredictor(
            checkpoint_path=os.path.join(config["NNINT_CKPT_DIR"], "nnInteractive_v1.0"),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        raise ValueError(f"Unknown method: {method}.")
                                   
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
            data = np.load(input_filepath, allow_pickle=True)
            gt_data = np.load(gt_filepath)

            image = data["imgs"]
            spacing = data["spacing"]
            gts = gt_data["gts"]
            print(image.shape)
            initial_bbox = data.get("boxes", None)  # Use .get for optional keys

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
                if bbox_input is not None : 
                    print("Input: BBox")
                if clicks_input is not None : 
                    print(f"Input: Clicks - {clicks_input}")

                # Run model inference
                current_segmentation, infer_time = predictor.predict(
                    image=image,
                    spacing=spacing,
                    bboxs=bbox_input,
                    clicks=clicks_input,
                    prev_pred=prev_prediction,  # Pass previous prediction
                    num_classes_max=num_classes_max,
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
            # TODO : log at every refinement step to see progress
            # TODO : Log the 3d seg ?
            if use_wandb:
                # get middle slice 
                wandb_img = image[
                    image.shape[0] // 2, :, :
                ]
                wandb_pred = current_segmentation[
                    current_segmentation.shape[0] // 2, :, :
                ]
                wandb_gt = gts[gts.shape[0] // 2, :, :]


                # Save the middle slice image

                wandb.log(
                    {
                        "Case/DSC_AUC": dsc_auc,
                        "Case/NSD_AUC": nsd_auc,
                        "Case/DSC_Final": dsc_final,
                        "Case/NSD_Final": nsd_final,
                        "Case/TotalRunningTime": total_inference_time,
                        "Case/AvgRunningTime": case_summary["AvgRunningTime"],
                        "case_name": case_name,  # Log case name for grouping
                        "Segmentation": wandb.Image(
                            wandb_img,
                            masks={
                                "predictions": {
                                    "mask_data": wandb_pred,
                                    # "class_labels": {0: "Background", 1: "Foreground"},
                                },
                                "ground_truth": {
                                    "mask_data": wandb_gt,
                                    # "class_labels": {0: "Background", 1: "Foreground"},
                                },
                            },
                        ),
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
        "-me",
        "--method",
        default="nnint",
        type=str,
        help="method used for segmentation, sammed3d or nnint",
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
        "-m",
        "--num_classes_max",
        default=None,
        type=int,
        help="Maximum number of classes to predict (None for all)",
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
    parser.add_argument(
        "-i",
        "--val_imgs_path",
        default=os.path.join(config["VAL_DIR"], "3D_val_npz"),
        type=str,
        help="Path to the validation images directory",
    )
    parser.add_argument(
        "-g",
        "--validation_gts_path",
        default=os.path.join(config["VAL_DIR"], "3D_val_gt_interactive_seg"),
        type=str,
        help="Path to the validation ground truth directory",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        default=os.path.join(config["RESULTS_DIR"], "nnInt"),
        type=str,
        help="Path to the results directory",
    )

    args = parser.parse_args()

    """
    
    args.val_imgs_path = os.path.join(config["VAL_DIR"], "3D_val_npz")
    args.validation_gts_path = os.path.join(
        config["VAL_DIR"], "3D_val_gt_interactive_seg"
    )
    args.save_path = os.path.join(config["RESULTS_DIR"], "sammed3d")
    """
    evaluate(
        method=args.method,
        img_dir=args.val_imgs_path,
        gt_dir=args.validation_gts_path,
        output_dir=args.save_path,
        num_clicks=args.num_clicks,
        num_cases=args.num_cases,
        num_classes_max=args.num_classes_max,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        verbose=args.verbose,
    )
