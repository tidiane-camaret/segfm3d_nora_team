# SegFM3D_nora_team

Repository for our participation in the Interactive 3D Segmentation Challenge (CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation).

## Evaluation Script

This script (`scripts/eval.py`) runs the evaluation protocol locally, mimicking the competition's iterative refinement process (bounding box + clicks). [See original script](https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/CVPR25_iter_eval.py)

**Fast Evaluation for Debugging:**

To run a quick test on a single case ( `-n 1`), with only the bounding box iteration (`-c 0`), predicting only the first class (`-m 1`), and without logging to Weights & Biases (`--no_wandb`):

```bash
# Ensure you are in an environment managed by uv (already present in Meta's workspace : /work/dlclarge2/ndirt-SegFM3D )
uv run python scripts/eval.py -c 0 -n 1 -m 1 --no_wandb
```

(Note: This currently uses the SAM-Med3D predictor by default. Add classes in the /src directory to use other predictors.)

### Baseline: SAM-Med3D

Implementation and evaluation logic related to the SAM-Med3D baseline model.

Resource Requirements (from original SAM-Med3D repo):

GPU: Requires significant VRAM (e.g., 16 GB for batch_size=2, 48 GB for batch_size=6 during training/fine-tuning). Inference requirements may differ.

CPU RAM: ~64 GB recommended.

Training Time (Example): ~0.8 GPU hours per epoch on the Coreset track dataset (indicative value).



### Setup & Running:

Some dependencies are pinned to specific versions or forks required by the CVPR25_3DFM branch of the original SAM-Med3D repository.

These should be handled by installing the locked requirements file.
Install Dependencies:

```bash
# Using uv package manager
uv pip install -r requirements.lock
```

