#!/bin/bash
# run with bash cluster/interactive_job.sh

#typical ressources for an interactive job
# 1 GPU, 16 CPU, 64GB RAM, 48h

srun -p ml_gpu-rtx2080 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=16 \
     --gres=gpu:1 \
     --mem=64G \
     --time=48:00:00 \
     --pty bash