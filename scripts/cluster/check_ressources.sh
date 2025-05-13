#!/bin/bash
# run using :
# srun -p ml_gpu-rtx2080 --pty bash cluster/check_resources.sh

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Tasks: $SLURM_NTASKS"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Memory:"
free -h

echo "CPU info:"
lscpu | grep "CPU(s):"

echo "GPU info (if available):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -L
fi


