#!/bin/bash
#SBATCH --job-name=fasc_masking
#SBATCH --output=outputs/fasc_masking-%A_%a.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-gpu=6
#SBATCH --nice=10000
#SBATCH --array=0-1

files=(
#    ../outputs/baselines/task2_random.csv
#    ../outputs/differential_expression/task2_DE_cell_types.csv
    ../outputs/ablation/task2_abs.csv
    ../outputs/expected_gradients/task2_absolute_sum_expected_grads.csv
)

tasks=(
    2
    2
#    2
#    2
)

source activate fa_base
python mask.py --csv ${files[$SLURM_ARRAY_TASK_ID]} --task ${tasks[$SLURM_ARRAY_TASK_ID]}
