#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100
#SBATCH --time=1-05:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
#SBATCH --array=1-10
# Output and error files
#SBATCH -o slurm_logs/fusion_plot_data_extraction/outlogs/job.%J.out
#SBATCH -e slurm_logs/fusion_plot_data_extraction/errlogs/job.%J.err
# Email notifications
#SBATCH --mail-type=ALL
# Resource requirements commands

# Activating conda
eval "$(conda shell.bash hook)"
conda activate uq-wq

export XLA_PYTHON_CLIENT_MEM_FRACTION=".80"; \
export WANDB_API_KEY=9b5357df452ab07579f57dc9db74d1382a58a001; \
srun $(head -n $SLURM_ARRAY_TASK_ID job_files/extraction_plot_data/jobs_fusion_tailored_runs.txt | tail -n 1)