#!/bin/bash

# Define parameter lists
noise_paths=("0" "4" "8" "12")
num_targets=("10" "25" "50")
sampling_noise=("" "noise14")
coverage=(1000 40)
Nsampling=(1000 100)

use_gene=(0 1)
use_gene_str=('0' '1')

# Loop over parameters and execute srun jobs sequentially
for noise_path in "${noise_paths[@]}"; do
    for num_target in "${num_targets[@]}"; do
        # Loop over coverage and Nsampling arrays using indices
        animal_gt="max_${noise_path}_r_t_${num_target}_n_"
        sbatch 3c_slurm_single_declare_eval_positions "$animal_gt"
        for i in "${!coverage[@]}"; do

            cov="${coverage[$i]}"
            sampling="${Nsampling[$i]}"
            sampling_str="${sampling_noise[$i]}"
            animal="max_${noise_path}_r_t_${num_target}_n_${sampling_str}"

            for j in "${!use_gene_str[@]}"; do
              integrate_gene="${use_gene[$j]}"
              integrate_gene_str="${use_gene_str[$j]}"
              nn_name="VAE_max_${noise_path}_r_t_${num_target}_n${sampling_str}_usegene${integrate_gene_str}"
              # Submit SLURM job
              sbatch 5_slurm_single_VAE_eval "$animal" "$nn_name" $integrate_gene
            done
            sbatch 5_slurm_single_WF_execution "$animal"
        done

    done
done
