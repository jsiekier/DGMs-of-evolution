#!/bin/bash

# Define parameter lists
noise_paths=("0" "4" "8" "12")
num_targets=("10" "25" "50")
sampling_noise=("" "noise14")
coverage=(1000 40)  # Fixed the syntax issue, this should be 1000 40
Nsampling=(1000 100)  # Fixed the syntax issue, this should be 1000 100

# Loop over parameters and execute srun jobs sequentially
for noise_path in "${noise_paths[@]}"; do
    for num_target in "${num_targets[@]}"; do
        # Loop over coverage and Nsampling arrays using indices
        animal_no_noise="max_${noise_path}_r_t_${num_target}_n_"
        for i in "${!coverage[@]}"; do
            cov="${coverage[$i]}"
            sampling="${Nsampling[$i]}"
            sampling_str="${sampling_noise[$i]}"

            # Define file paths based on parameters
            animal="max_${noise_path}_r_t_${num_target}_n_${sampling_str}"

            sbatch 3b_slurm_single_s_estimation "$animal" "$animal_no_noise"
        done
    done
done
