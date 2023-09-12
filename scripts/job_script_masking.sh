#!/bin/bash
#SBATCH --job-name=fasc_masking
#SBATCH --output=outputs/fasc_masking-%A_%a.out
#SBATCH --error=outputs/fasc_masking-%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-gpu=6
#SBATCH --nice=10000
#SBATCH --array=0-9

files=(
    ../outputs/baselines/task1_random.csv
    ../outputs/baselines/task1_mean.csv
    ../outputs/differential_expression/task1_DE_control.csv
    ../outputs/differential_expression/task1_DE_rest.csv
    ../outputs/expected_gradients/task1_absolute_expected_grads_v2.csv
    ../outputs/integrated_gradients/task1_absolute_integrated_grads_v2.csv
    ../outputs/baselines/task2_random.csv
    ../outputs/differential_expression/task2_DE_cell_types.csv
    ../outputs/ablation/task2_abs.csv
    ../outputs/expected_gradients/task2_absolute_sum_expected_grads.csv
)

tasks=(
    1
    1
    1
    1
    1
    1
    2
    2
    2
    2
)

source activate fa_base
python mask.py --csv ${files[$SLURM_ARRAY_TASK_ID]} --task ${tasks[$SLURM_ARRAY_TASK_ID]}