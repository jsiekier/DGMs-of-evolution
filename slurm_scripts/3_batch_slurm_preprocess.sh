#!/bin/bash

# Define parameter lists
noise_paths=("0" "4" "8" "12")
num_targets=("10" "25" "50")

# Loop over parameters and submit SLURM jobs
for noise_path in "${noise_paths[@]}"; do
    for num_target in "${num_targets[@]}"; do
        animal="max_${noise_path}_r_t_${num_target}_n_"

        # Submit SLURM job
        sbatch 3_slurm_single_preprocess "$animal"
    done
done
