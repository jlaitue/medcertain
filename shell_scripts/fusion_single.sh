#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-05:00:00
#SBATCH --cpus-per-task=20
# Output and error files
#SBATCH -o slurm_logs/fusion_psvi_3_runs/outlogs/job.%J.out
#SBATCH -e slurm_logs/fusion_psvi_3_runs/errlogs/job.%J.err
# Email notifications
#SBATCH --mail-type=ALL
# Resource requirements commands

# Activating conda
eval "$(conda shell.bash hook)"
conda activate uq-wq

export XLA_PYTHON_CLIENT_MEM_FRACTION=".80"; \
export WANDB_API_KEY=9b5357df452ab07579f57dc9db74d1382a58a001; \

python trainer.py \
--config configs/nn-tdvi-pt-fusion-mimic-psvi.json \
--jobid 499 --pretrained_prior --save_to_wandb --model_for_final_eval LAST \
--label_file_splits medfuse_test --learning_rate 0.0006465 --batch_size 16 \
--num_epochs 5 --alpha 0 --context_batch_size 32 --context_points merged \
--prior_var 1000 --prior_likelihood_scale 0.1 --prior_likelihood_f_scale 1 \
--prior_likelihood_cov_scale 0.1 --prior_likelihood_cov_diag 5 \
--mimic_task in-hospital-mortality --wandb_project uq-fusion-psvi-3_1-merged-psvi1-mortality \
--pretrained_prior --pretrained_prior_path checkpoints_best_models/929203_Fusion_DET_20_0.0064659501030560436_16_0.0_1_exprt_394_seed_4_in-hospital-mortality_medfuse_test_20/checkpoint_20 \
--context_points_data_file Context-III/in-hospital-mortality/context_set_3_cos_sim_type1_in-hospital-mortality_1_std.npz --seed 4