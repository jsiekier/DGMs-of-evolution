#!/bin/bash

# Define parameter lists
noise_paths=("0" "4" "8" "12")
num_targets=("10" "25" "50")

# Loop over parameters and submit SLURM jobs
for noise_path in "${noise_paths[@]}"; do
    for num_target in "${num_targets[@]}"; do
        haplo_file="data/haplos_max_${noise_path}_r"
        target_file="targets/targets_max${noise_path}_t_${num_target}.txt"
        sync_zip="/lustre/miifs01/project/m2_datamining/sync_data/haplos_max_${noise_path}_r_t_${num_target}.sync"
        sync_unz="/lustre/miifs01/project/m2_datamining/sync_data/haplos_max_${noise_path}_r_t_${num_target}.unz.sync"

        # Submit SLURM job
        sbatch slurm_single_simulate "$haplo_file" "$target_file" "$sync_zip" "$sync_unz"
    done
done
