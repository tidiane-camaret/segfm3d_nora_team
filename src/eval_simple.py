"""
Main evaluation script
Adapted from https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/CVPR25_iter_eval.py
to run locally
- without the need of docker
- without eval order constraints that are specific to the competition
The rest of the logic should stay the same
"""

import argparse
import os
import random
import sys
import traceback
from collections import OrderedDict

import cc3d
import numpy as np
import torch
import wandb
from scipy import integrate
from src.config import config
from src.eval_metrics import (  # TODO : Use the competition repo as source instead
    compute_edt,
    compute_multi_class_dsc,
    compute_multi_class_nsd,
    sample_coord,
)
from tqdm import tqdm

random.seed(42)


def evaluate(
    method,
    img_dir,  # input images
    gt_dir,  # input ground truth
    output_dir,
    n_clicks=5,
    n_cases=10,
    n_classes_max=None,  # limiting the number of classes to evaluate for speedup (sometimes 15 classes)
    use_wandb=True,
    wandb_project="segfm3d_nora_team",
    verbose=False,
    save_segs=True,
    checkpoint_path=None,  # used in nnint_custom
):
    torch.set_grad_enabled(False)  # Disable gradient calculation for inference
    if save_segs:
        print(
            "Warning: Saving segmentations is enabled. Will take more time and space."
        )
    ### Logging via wandb ###

    if use_wandb:
        # Initialize WandB
        wandb.init(
            project=wandb_project,
            config={
                "img_dir": img_dir,
                "gt_dir": gt_dir,
                "num_clicks": n_clicks,
                "evaluation_mode": "local_script",
                "gpu_available": torch.cuda.is_available(),
                "method": method,
                "checkpoint_name": os.path.basename(checkpoint_path) if checkpoint_path else None,
            },
        )
        # Define metrics for WandB summary
        wandb.define_metric("Case/DSC_AUC", summary="mean")
        wandb.define_metric("Case/NSD_AUC", summary="mean")
        wandb.define_metric("Case/DSC_Final", summary="mean")
        wandb.define_metric("Case/NSD_Final", summary="mean")
        wandb.define_metric("Case/TotalRunningTime", summary="mean")
        wandb.define_metric("Iteration")
        wandb.define_metric("DSC", step_metric="Iteration")
        wandb.define_metric("NSD", step_metric="Iteration")
        wandb.define_metric("RunningTime", step_metric="Iteration")

    ### Load the method ###
    print("device that will be used", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if method == "sammed3d":
        from src.methods.sammed3d import SAMMed3DPredictor

        predictor = SAMMed3DPredictor(checkpoint_path=config["SAM_CKPT_PATH"])
    elif method == "nnint":
        from src.methods.nninteractive import nnInteractivePredictor

        predictor = nnInteractivePredictor(
            checkpoint_path=os.path.join(
                config["NNINT_CKPT_DIR"], "nnInteractive_v1.0"
            ),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=verbose,
        )

    elif method == "nnint_core":
        from src.methods.nninteractivecore import nnInteractiveCorePredictor

        predictor = nnInteractiveCorePredictor(
            checkpoint_path=os.path.join(
                config["NNINT_CKPT_DIR"], "nnInteractive_v1.0"
            ),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=verbose,
        )
    elif method == "nnint_orig":
        from src.methods.nninteractiveorig import nnInteractiveOrigPredictor

        predictor = nnInteractiveOrigPredictor(
        checkpoint_path=os.path.join(
            config["NNINT_CKPT_DIR"], "may_25/nnInteractiveTrainer_CVPR2025"
        ),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=verbose,
    )
    elif method == "nnint_custom":
        from src.methods.nninteractiveorig import nnInteractiveOrigPredictor

        predictor = nnInteractiveOrigPredictor(
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=verbose,
        )
    elif method == "simple":
        from src.method import SimplePredictor
        trained_checkpoint_path = '/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_segfm-robin/data/model-checkpoints/new-dataset/fold_0/checkpoint_4_2025-05-30_12-27-37-246.pth'#'/nfs/data/nii/data1/Analysis/GPUnet/ANALYSIS_segfm-robin/data/model-checkpoints/new-dataset/fold_0/checkpoint_5_2025-05-30_07-44-18-169.pth'
        predictor = SimplePredictor(trained_checkpoint_path, device="cuda", include_previous_clicks=True, n_pred_iters=3)
    else:
        raise ValueError(f"Unknown method: {method}.")

    ### List cases (npz files) ###
    output_dir = os.path.join(output_dir, method)
    
    cases = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])

    """
    processed_cases = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])
    remaining_cases = sorted(list(set(cases) - set(processed_cases)))

    cases = remaining_cases
    """
    # just now start at specific image to check error
    
    # randomize which cases are evaluated
    np.random.shuffle(cases)
    #cases = cases[7:n_cases+7] if n_cases > 0 else cases  # limit number of cases to evaluate
    cases = cases[:n_cases] if n_cases > 0 else cases


    if len(cases) == 0:
        print("No cases found in the input directory.")
        print(img_dir)
        return
    print(f"Cases to evaluate: {len(cases)}")

    os.makedirs(output_dir, exist_ok=True)
    metric = OrderedDict(
        {
            "CaseName": [],
            "TotalRunningTime": [],
            "RunningTime_1": [],
            "RunningTime_2": [],
            "RunningTime_3": [],
            "RunningTime_4": [],
            "RunningTime_5": [],
            "RunningTime_6": [],
            "DSC_AUC": [],
            "NSD_AUC": [],
            "DSC_Final": [],
            "NSD_Final": [],
            "DSC_1": [],
            "DSC_2": [],
            "DSC_3": [],
            "DSC_4": [],
            "DSC_5": [],
            "DSC_6": [],
            "NSD_1": [],
            "NSD_2": [],
            "NSD_3": [],
            "NSD_4": [],
            "NSD_5": [],
            "NSD_6": [],
            "num_class": [],
            "runtime_upperbound": [],
        }
    )

    ### Loop through cases ###

    for case_filename in tqdm(cases, desc="Evaluating Cases"):
        print(f"Evaluating case: {case_filename}")

        case_name = os.path.splitext(case_filename)[0]
        input_filepath = os.path.join(img_dir, case_filename)
        gt_filepath = os.path.join(gt_dir, case_filename)

        metric_temp = {}
        real_running_time = 0
        dscs = []
        nsds = []
        all_segs = []

        if not os.path.exists(gt_filepath):
            print(f"Warning: Ground truth file not found for {case_name}. Skipping.")
            continue

        try:
            ### Load data ###
            data = np.load(input_filepath, allow_pickle=True)
            gt_data = np.load(gt_filepath)

            image = data["imgs"]
            spacing = data["spacing"]
            gts = gt_data["gts"]
            boxes = data.get("boxes", None)

            segs = np.zeros_like(gts, dtype=np.uint8)  # Initialize with zeros

            if np.any(np.array(image.shape) > 1000):
                print(
                    f"Warning: One of the dims is > 1000 for {case_name}. shape : {image.shape}. Skipping because of memory constraints. Seg segs = 0"
                )
                if save_segs:
                    np.savez_compressed(
                        os.path.join(output_dir, case_name),
                        segs=segs,
                        all_segs=[segs],  # store all intermediate predictions
                    )


            ### Initialize interaction objects and metrics ###
            unique_gts = np.sort(np.unique(gts))
            n_classes = len(unique_gts) - 1  # Exclude background class 0
            clicks_cls = [
                {"fg": [], "bg": []} for _ in range(n_classes)
            ]  # skip background class 0
            clicks_order = [[] for _ in range(n_classes)]

            # Initialize segmentation

            for it in range(n_clicks + 1):  # + 1 due to bbox pred at iteration 0

                if it == 0:
                    ### Checking if box exists ###
                    if boxes is None:
                        if verbose:
                            print(
                                f"This sample does not use a Bounding Box for the initial iteration {it}"
                            )
                        metric_temp["RunningTime_1"] = 0
                        metric_temp["DSC_1"] = 0
                        metric_temp["NSD_1"] = 0
                        dscs.append(0)
                        nsds.append(0)
                        continue

                    if verbose:
                        print(f"Using Bounding Box for iteration {it}")

                else:
                    all_segs.append(segs.astype(np.uint8))
                    ### Generating clicks from previous segmentation ###
                    for ind, cls in enumerate(sorted(unique_gts[1:])):
                        if cls == 0:
                            continue  # skip background

                        segs_cls = (segs == cls).astype(
                            np.uint8
                        )  ### TODO : the segs are not defined yet
                        gts_cls = (gts == cls).astype(np.uint8)

                        # Compute error mask
                        error_mask = (segs_cls != gts_cls).astype(np.uint8)
                        if np.sum(error_mask) > 0:
                            errors = cc3d.connected_components(
                                error_mask, connectivity=26
                            )  # 26 for 3D connectivity

                            # Calculate the sizes of connected error components
                            component_sizes = np.bincount(errors.flat)

                            # Ignore non-error regions
                            component_sizes[0] = 0

                            # Find the largest error component
                            largest_component_error = np.argmax(component_sizes)

                            # Find the voxel coordinates of the largest error component
                            largest_component = errors == largest_component_error

                            edt = compute_edt(largest_component)
                            center = sample_coord(edt)

                            if (
                                gts_cls[center] == 0
                            ):  # oversegmentation -> place background click
                                assert segs_cls[center] == 1
                                clicks_cls[ind]["bg"].append(list(center))
                                clicks_order[ind].append("bg")
                            else:  # undersegmentation -> place foreground click
                                assert segs_cls[center] == 0
                                clicks_cls[ind]["fg"].append(list(center))
                                clicks_order[ind].append("fg")

                            assert largest_component[center]  # click within error

                            if verbose:
                                print(
                                    f"Class {cls}: Largest error component center is at {center}"
                                )
                        else:
                            if verbose:
                                print(
                                    f"Class {cls}: No error connected components found. Prediction is perfect! No clicks were added."
                                )

                ### Model prediction ###

                segs, prediction_metrics = predictor.predict(
                    image=image,
                    # spacing=spacing,
                    bboxs=boxes,
                    clicks=(clicks_cls, clicks_order),
                    # is_bbox_iteration=it == 0,
                    prev_pred=segs,  # Pass previous prediction
                    num_classes_max=n_classes_max,
                )

                ### Computing Metrics

                real_running_time += prediction_metrics["infer_time"]
                metric_temp[f"RunningTime_{it + 1}"] = prediction_metrics["infer_time"]

                dsc = compute_multi_class_dsc(gts, segs)
                # compute nsd
                if dsc > 0.2:
                    # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
                    nsd = compute_multi_class_nsd(gts, segs, spacing)
                else:
                    nsd = 0.0  # Assume model performs poor on this sample
                dscs.append(dsc)
                nsds.append(nsd)
                metric_temp[f"DSC_{it + 1}"] = dsc
                metric_temp[f"NSD_{it + 1}"] = nsd
                print("Dice", dsc, "NSD", nsd)
                if use_wandb:
                    wandb.log(
                        {
                            "Iteration": it,
                            "Case": case_name,
                            "DSC": dscs[it] if it < len(dscs) else 0,
                            "NSD": nsds[it] if it < len(nsds) else 0,
                            "RunningTime": (
                                metric_temp[f"RunningTime_{it + 1}"]
                                if f"RunningTime_{it + 1}" in metric_temp
                                else 0
                            ),
                            "Forward pass count": prediction_metrics["forward_pass_count"]
                        }
                    )

            all_segs.append(segs.astype(np.uint8))


            try:
                dsc_auc = integrate.cumulative_trapezoid(
                    np.array(dscs[-n_clicks:]), np.arange(n_clicks)
                )[
                    -1
                ]  # AUC is only over the point prompts since the bbox prompt is optional
                nsd_auc = integrate.cumulative_trapezoid(
                    np.array(nsds[-n_clicks:]), np.arange(n_clicks)
                )[-1]
            except Exception as e:
                print(
                    f"Error calculating AUC: {e}. Using last DSC and NSD values instead."
                )
                dsc_auc = dscs[-1]
                nsd_auc = nsds[-1]

            dsc_final = dscs[-1]
            nsd_final = nsds[-1]
            for k, v in metric_temp.items():
                metric[k].append(v)
            metric["CaseName"].append(case_name)
            metric["TotalRunningTime"].append(real_running_time)
            metric["DSC_AUC"].append(dsc_auc)
            metric["NSD_AUC"].append(nsd_auc)
            metric["DSC_Final"].append(dsc_final)
            metric["NSD_Final"].append(nsd_final)

            # Save the metric file to output_dir
            """ 
            metric_df = pd.DataFrame(metric) -> Error in /software/anaconda3/envs/segfm3d_2/lib/python3.12/site-packages/pandas/core/internals/construction.py, line 677, in _extract_index: All arrays must be of the same length
            TODO : Look at the dataframe
            
            
            metric_df.to_csv(
                os.path.join(output_dir, "norateam_metrics.csv"), index=False
            )
            """

            if save_segs:
                np.savez_compressed(
                    os.path.join(output_dir, case_name),
                    segs=segs,
                    all_segs=all_segs,  # store all intermediate predictions
                )

            if use_wandb:
                # get middle slice
                wandb_img = image[image.shape[0] // 2, :, :]
                wandb_pred = segs[segs.shape[0] // 2, :, :]
                wandb_gt = gts[gts.shape[0] // 2, :, :]

                # Save the middle slice image

                wandb.log(
                    {
                        "Case/DSC_AUC": dsc_auc,
                        "Case/NSD_AUC": nsd_auc,
                        "Case/DSC_Final": dsc_final,
                        "Case/NSD_Final": nsd_final,
                        "Case/TotalRunningTime": real_running_time,
                        "Case_name": case_name,
                        "Case/NumClasses": n_classes,
                        "Case/NumClassesUsed": n_classes_max,
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
            traceback.print_exc()
            print(f"Error message: {e}")
    return metric


# --- Argparse and Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Local Interactive Segmentation Evaluation")
    parser.add_argument(
        "-m", "--method", default="nnint", type=str, help="method used for segmentation"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=os.path.join(config["RESULTS_DIR"]),
        type=str,
        help="Local segmentation and metrics output path",
    )
    parser.add_argument(
        "-ca",
        "--n_cases",
        default=10,
        type=int,
        help="Number of cases to evaluate. If 0, all cases will be evaluated",
    )
    parser.add_argument(
        "-cl",
        "--n_clicks",
        default=5,
        type=int,
        help="Number of click refinement iterations",
    )
    parser.add_argument(
        "-ncm",
        "--n_classes_max",
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
        "--save_segs", action="store_true", help="Save all segmentations"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output for clicks etc."
    )
    parser.add_argument(
        "--img_dir",
        default=os.path.join(config["DATA_DIR"], "3D_val_npz"),
        type=str,
        help="Path to the validation images",
    )
    parser.add_argument(
        "--gt_dir",
        default=os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg"),
        type=str,
        help="Path to the validation ground truth",
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Path to the model checkpoint. used in nnint_custom",
    )

    args = parser.parse_args()

    evaluate(
        method=args.method,
        img_dir=args.img_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        n_clicks=args.n_clicks,
        n_cases=args.n_cases,
        n_classes_max=args.n_classes_max,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        verbose=args.verbose,
        save_segs=args.save_segs,
        checkpoint_path=args.checkpoint_path,  # used in nnint_custom
    )
