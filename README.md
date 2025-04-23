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

## Evaluating a submission
### Converting Docker image to Singularity
The [evaluation script](CVPR-MedSegFMCompetition/CVPR25_iter_eval.py) is designed to run in a Docker container. We don't have the ability to use Docker in the cluster, but we can use Singularity. 

```bash
# Convert the Docker image to a Singularity image
singularity build docker_submission/images/sammed3d_baseline.sif docker-archive://docker_submission/images/sammed3d_baseline.tar

# Test the Singularity image (nv : GPU usage)
singularity shell --nv docker_submission/images/sammed3d_baseline.sif

# bind the input and output directories
singularity shell --nv -B $PWD/data:/workspace/inputs,$PWD/results/sammed3d:/workspace/outputs  docker_submission/images/sammed3d_baseline.sif 


#Then sh predict.sh
```

### Running the Evaluation Script
Modifiy the evaluation script so that it uses Singularity instead of Docker :
```bash
# Original Docker command:
# cmd = 'docker container run --gpus "device=0" -m 32G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)

# New Singularity command:
if torch.cuda.is_available(): # GPU available
    cmd = 'singularity run --nv -B $PWD/inputs/:/workspace/inputs/,$PWD/outputs/:/workspace/outputs/ {}.sif /bin/bash -c "sh predict.sh"'.format(teamname)
else:
    cmd = 'singularity run -B $PWD/inputs/:/workspace/inputs/,$PWD/outputs/:/workspace/outputs/ {}.sif /bin/bash -c "sh predict.sh"'.format(teamname)
```
Also, remove Remove "docker image load" and "docker rmi" commands from the script.

## Building Docker images

Use the following command to build the image via Singularity:

```bash
# Build the Docker image using Singularity
singularity build sam-med3d.sif docker://sam-med3d:latest
# Save the Docker image to a tar file
singularity save sam-med3d.sif -o sam-med3d.tar
```