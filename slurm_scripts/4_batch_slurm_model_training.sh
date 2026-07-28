#!/bin/bash
# Define parameter lists
noise_paths=("0" "4" "8" "12")
num_targets=("10" "25" "50")
sampling_noise=("" "noise14")
coverage=(1000 40)
Nsampling=(1000 100)

use_gene=(0 1)
use_gene_str=('0' '1')

time_batch_size=6



data_transform='None'
enc1_activation='tanh'
use_enc3=1
use_neg_softmax=2
num_train_it=0
KLD=4
free_bit=0
NLL_lambda=0
z_dim=10
n_neighbors=50
include_parallel_info=0
enc_type='sim'



# Loop over parameters and execute srun jobs sequentially
for noise_path in "${noise_paths[@]}"; do
    for num_target in "${num_targets[@]}"; do
        # Loop over coverage and Nsampling arrays using indices
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
              sbatch 4_slurm_single_VAE_train "$animal" "$nn_name" "$integrate_gene" $time_batch_size $KLD $free_bit $NLL_lambda $z_dim $n_neighbors "$data_transform" "$enc1_activation" $use_enc3 $use_neg_softmax $num_train_it $include_parallel_info "$enc_type"

              done
        done
    done
done
