#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100
#SBATCH --time=1-05:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
# Output and error files
#SBATCH -o slurm_logs/unimodal/outlogs/job.%J.out
#SBATCH -e slurm_logs/unimodal/errlogs/job.%J.err
# Email notifications
#SBATCH --mail-type=ALL
# Resource requirements commands

# Activating conda
eval "$(conda shell.bash hook)"
conda activate uq-wq

export XLA_PYTHON_CLIENT_MEM_FRACTION=".80"; \
export WANDB_API_KEY=9b5357df452ab07579f57dc9db74d1382a58a001; \
python trainer.py --config configs/nn-tdvi-pt-ehr-mimic.json \
--save_to_wandb --jobid 001 --label_file_splits medfuse_test \
--model_for_final_eval LAST --batch_size 16 --num_epochs 50 \
--learning_rate 2.951e-05 --mimic_task phenotyping --wandb_project uq-uni-ehr --seed 0