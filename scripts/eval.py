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

from src.config import config
from src.eval import evaluate

random.seed(42)


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
