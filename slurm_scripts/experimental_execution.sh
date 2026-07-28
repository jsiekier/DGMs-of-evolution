#!/bin/bash


animal='experimental'
#sync_unz='/lustre/miifs01/project/m2_datamining/sync_data/Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID_2.sync'
#sbatch 3_slurm_single_preprocess "$animal"
#sbatch 3c_slurm_single_declare_eval_positions "$animal"

#srun Rscript estimate_Ne.R --filename "$sync_unz" --animal "$animal"
#sbatch 3b_slurm_single_s_estimation "$animal" "$animal" --num_train_it 0

#sbatch 5_slurm_single_WF_execution "$animal"

animals=('experimental' 'experimental')
use_gene=(0 1)
use_gene_str=('0' '1')

for i in "${!animals[@]}"; do

    data_transform='None' 
    enc1_activation='tanh'
    use_enc3=1
    use_neg_softmax=2 #"${use_neg_softmaxs[$i]}"
    time_batch_size=4 #"${time_batch_sizes[$i]}" #6 #4 #6 #
    num_train_it=0

    integrate_gene="${use_gene[$i]}" #0

    KLD=4
    free_bit=0 
    NLL_lambda=0 
    z_dim=10 #10 #

    n_neighbors=50 
    include_parallel_info=0
    enc_type='sim'

    animal="${animals[$i]}"

    integrate_gene_str="${use_gene_str[$i]}"

    nn_name="VAE_gene${integrate_gene_str}_tbs${time_batch_size}_n100_negsoft${use_neg_softmax}_z10_1enc3_${animal}" #"VAE_gene1_tbs6_n100_${animal}"  #"VAE_gene1_l40_n100_${animal}"  #

    #sbatch 4_slurm_single_VAE_train "$animal" "$nn_name" "$integrate_gene" $time_batch_size $KLD $free_bit $NLL_lambda $z_dim $n_neighbors "$data_transform" "$enc1_activation" $use_enc3 $use_neg_softmax $num_train_it $include_parallel_info "$enc_type"
    #sbatch 5_slurm_single_VAE_eval "$animal" "$nn_name" $integrate_gene $KLD #$free_bit $NLL_lambda


    done

