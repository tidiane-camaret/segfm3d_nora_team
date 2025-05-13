#!/bin/bash
# run using :
# sbatch cluster/batch_job.sh


#SBATCH -p ml_gpu-rtx2080      # Partition name
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=16     # CPUs per task
#SBATCH --mem=64G              # Memory per node
#SBATCH --gres=gpu:1           # GPUs per node
#SBATCH --time=48:00:00         # Time limit
#SBATCH --job-name=segfm3d_nnint      # Job name
#SBATCH --output=results/cluster/job_%j.out    # Standard output file (%j will be replaced by job ID)
#SBATCH --error=results/cluster/job_%j.err     # Standard error file

# Your commands to run go here
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

cd /work/dlclarge2/ndirt-SegFM3D/segfm3d_nora_team

uv run scripts/eval.py --save_segs -ca 0

echo "Job finished at $(date)"