# SegFM3D_nora_team

Interactive 3d segmentation challenge

# Evaluation script 
## fast eval for debugging
uv run python scripts/eval.py -c 1 -n 1

## Baselines stats
### SAM-Med3D 
GPU: 16 GB DRAM (batch_size=2), 48 GB DRAM (batch_size=6)
CPU : 64 GB
compute for 1 epoch : 0.8 GPU hours (Coreset track)

some dependecies (see https://github.com/uni-medical/SAM-Med3D/tree/CVPR25_3DFM)
uv pip install git+https://github.com/uni-medical/MedIM.git@CVPR25_3DFM

uv run scripts/baselines/eval_sammed3d.py