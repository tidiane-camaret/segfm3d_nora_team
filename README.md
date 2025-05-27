# SegFM3D_nora_team

Repository for our participation in the Interactive 3D Segmentation Challenge (CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation).

## Local evaluation Script

This script (`scripts/eval.py`) runs the evaluation protocol locally, mimicking the competition's iterative refinement process (bounding box + clicks). [See original script](https://github.com/JunMa11/CVPR-MedSegFMCompetition/blob/main/CVPR25_iter_eval.py)

**Fast Evaluation for Debugging:**

To run a quick test on a single case ( `-ca 1`), with only the bounding box iteration (`-cl 0`), predicting only the first class (`-ncm 1`), and without logging to Weights & Biases (`--no_wandb`):

```bash
# Ensure you are in an environment managed by uv (already present in Meta's workspace : /work/dlclarge2/ndirt-SegFM3D )
uv run python scripts/eval.py -ca 1 -cl 1 -ncm 1 --no_wandb
```



## Methods  

```--method``` tag

```nnint```
based on the [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) framework. It uses a pre-trained nnUNet model as the backbone and implements the interactive segmentation process using bounding boxes and clicks.

```nnintcore```
a simplified version of nnint, using the backbone only, without zoom-in and zoom-out features.

```sammed3d```
based on the [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D), which is a 3D extension of the Segment Anything Model (SAM). It uses the SAM model as the backbone and implements the interactive segmentation process using bounding boxes and clicks.

## Submission
The official evaluation script is designed to run in a Docker container. If you don't have Docker installed, you can use Singularity as an alternative (see below).

### Using Docker : 

```bash
### Building the Docker image

cd docker_context/
docker build -t norateam:latest .

### Testing the Docker image on one case (nv : GPU usage)
# Note : The official script copies the images in a temp directory first. 
# We reproduce this behavior by providing one image in docker_submission/test/inputs/ 

cd docker_submission/ 

docker container run --gpus "device=0" -m 32G --name norateam --rm -v $PWD/test/inputs/:/workspace/inputs/ -v $PWD/test/outputs/:/workspace/outputs/ norateam:latest /bin/bash -c "sh predict.sh"  

### Saving as a tar file

docker save norateam:latest | gzip > /nfs/norasys/notebooks/camaret/segfm3d_nora_team/docker_images/submission/norateam.tar.gz

### Evaluating the image using the official script 

# /data contains sample image and gt

python /nfs/norasys/notebooks/camaret/cvpr25/CVPR-MedSegFMCompetition/CVPR25_iter_eval.py --docker_folder /nfs/norasys/notebooks/camaret/segfm3d_nora_team/docker_images/submission --test_img_path /nfs/norasys/notebooks/camaret/segfm3d_nora_team/docker_submission/data/inputs/3D_val_npz --save_path /nfs/norasys/notebooks/camaret/segfm3d_nora_team/docker_submission/data/outputs --validation_gts_path /nfs/norasys/notebooks/camaret/segfm3d_nora_team/docker_submission/data/inputs/3D_val_gt --verbose

# with more data (cvpr25/data/)
python /nfs/norasys/notebooks/camaret/cvpr25/CVPR-MedSegFMCompetition/CVPR25_iter_eval.py --docker_folder /nfs/norasys/notebooks/camaret/segfm3d_nora_team/docker_images/submission --test_img_path /nfs/norasys/notebooks/camaret/cvpr25/data/3D_val_npz --save_path /nfs/norasys/notebooks/camaret/segfm3d_nora_team/docker_submission/data/outputs --validation_gts_path /nfs/norasys/notebooks/camaret/cvpr25/data/3D_val_gt/3D_val_gt_interactive --verbose


# evaluating on the test examples (/nfs/norasys/notebooks/camaret/cvpr25/test_demo)
cd /nfs/norasys/notebooks/camaret/cvpr25/test_demo
docker container run --gpus "device=0" -m 32G --name norateam --rm -v $PWD/imgs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ norateam:latest /bin/bash -c "sh predict.sh"
```

### Optimizing the docker container
Install docker-squash:
´pip install docker-squash´
Install docker-slim:
´curl -sL https://raw.githubusercontent.com/slimtoolkit/slim/master/scripts/install-slim.sh | sudo -E bash -´

Combine Run commands, don't create pip cache when installing (reduces uncompressed image by 3GB):
´docker build -f DockerfileOptimized -t norateam:latest .´
Make sure we have a single layer:
´docker-squash -t norateam:latest norateam:latest´
Run the container, check what files are getting used and throw all others out (big gain in size reduction, untested):
´slim build --target norateam:latest --tag norateam:latest --http-probe=false --include-workdir --mount $PWD/test/inputs/:/workspace/inputs/ --mount $PWD/test/outputs/:/workspace/outputs/ --exec "sh predict.sh"´

Afterwards test and save image like before.


### Using Singularity : 
#### Converting Docker image to Singularity

```bash
# Convert the Docker image to a Singularity image
singularity build docker_images/baselines/nninteractive_alldata.sif docker-archive://docker_images/baselines/nninteractive_alldata.tar

# Test the Singularity image (nv : GPU usage)
singularity shell --nv docker_images/baselines/singularity/nninteractive_alldata.sif

# bind the input and output directories
singularity shell --nv -B $PWD/data:/workspace/inputs,$PWD/results/sammed3d:/workspace/outputs  docker_images/baselines/singularity/nninteractive_alldata.sif 


#Then sh predict.sh
```

#### Running the Evaluation Script
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

```bash
python ../scripts/CVPR25_iter_eval_singularity.py --docker_folder ../docker_images/baselines/singularity --test_img_path ../data/3D_val_npz --save_path ../results/nnint_alldata_baseline --validation_gts_path ../data/3D_val_gt_interactive_seg --verbose
```

#### Building Docker images from Singularity


```bash
# Build the Docker image using Singularity
singularity build sam-med3d.sif docker://sam-med3d:latest
# Save the Docker image to a tar file
singularity save sam-med3d.sif -o sam-med3d.tar
```
