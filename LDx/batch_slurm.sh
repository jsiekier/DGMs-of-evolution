#!/bin/bash

# Define parameter lists
noise_paths=("0" "4" "8" "12")
num_targets=("10")

# Loop over parameters and submit SLURM jobs
for noise_path in "${noise_paths[@]}"; do
    for num_target in "${num_targets[@]}"; do
        haplo_file="simulations/data/haplos_max_${noise_path}_r"
        out_file="sync_data/gen0_fasta/gen0_max${noise_path}.fasta"
        # Submit SLURM job
        out_file1="sync_data/gen0_fasta/gen0_max${noise_path}sim_R1.fq"
        out_file2="sync_data/gen0_fasta/gen0_max${noise_path}sim_R2.fq"
        sim_sam="sync_data/gen0_fasta/gen0_max${noise_path}sim.sam"
        sim_sorted="sync_data/gen0_fasta/gen0_max${noise_path}sim.sorted.bam"
        sim_mpileup="sync_data/gen0_fasta/gen0_max${noise_path}sim.mpileup"
        sim_sync="sync_data/gen0_fasta/gen0_max${noise_path}sim.sync"
        out_file_sampled="sync_data/gen0_fasta/gen0_max_haplos_sampled${noise_path}"
        vcf="sync_data/gen0_fasta/gen0_max_haplos_sampled${noise_path}.vcf.gz"
        ldx_out="sync_data/gen0_fasta/gen0_max_haplos_sampled${noise_path}.txt"
        tmp_bam="sync_data/gen0_fasta/gen0_max_haplos_sampled${noise_path}tmp.bam"
        tmp_fixmate="sync_data/gen0_fasta/gen0_max_haplos_sampled${noise_path}.fixmate.bam"

        sbatch slurm_single_simulate "$haplo_file" "$out_file" "$out_file1" "$out_file2" "$sim_sam" "$sim_sorted" "$sim_mpileup" "$sim_sync" "$out_file_sampled" "$vcf" "$ldx_out" "$tmp_bam" "$tmp_fixmate"
    done
done
