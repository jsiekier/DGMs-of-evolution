#!/bin/bash

# Define parameter lists
noise_paths=("2" "4" "6" "7" "9") #
chrom="2L" #2L 2R 3R 3L X

# Loop over parameters and submit SLURM jobs
for noise_path in "${noise_paths[@]}"; do

    haplo_file="simulations/data/"
    out_file="/LDx/vcf/${noise_path}_${chrom}variants.filtered.vcf.gz"
    # Submit SLURM job
    in_file1="LDx/fastq/Base_${noise_path}/Base_${noise_path}_R1.fastq.gz"
    in_file2="LDx/fastq/Base_${noise_path}/Base_${noise_path}_R2.fastq.gz"

    sim_dedup="LDx/bam_files/${noise_path}mapped.dedup.bam"
    sim_sorted="LDx/bam_files/${noise_path}sim.sorted.bam"
    sim_sam="LDx/sam/${noise_path}_${chrom}.dedup.sorted.sam"
    sim_sync="/sync_data/gen0_fasta/gen0_max${noise_path}sim.sync"
    out_file_sampled="sync_data/gen0_fasta/gen0_max_haplos_sampled${noise_path}"
    vcf="LDx/vcf/${noise_path}_${chrom}variants.vcf.gz"
    ldx_out="LDx/ldx_out/${noise_path}_${chrom}_filter.txt"
    tmp_bam="LDx/bam_files/${noise_path}tmp.bam"
    tmp_fixmate="sync_data/gen0_fasta/gen0_max_haplos_sampled${noise_path}.fixmate.bam"
    out_file1_trim="LDx/fast_q/trim_ERR${noise_path}_1.fastq.gz"
    out_file2_trim="LDx/fast_q/trim_ERR${noise_path}_2.fastq.gz"


    sbatch slurm_LDx "$haplo_file" "$out_file" "$in_file1" "$in_file2" "$sim_dedup" "$sim_sorted" "$sim_sam" "$sim_sync" "$out_file_sampled" "$vcf" "$ldx_out" "$tmp_bam" "$tmp_fixmate" "$out_file1_trim" "$out_file2_trim" "$chrom"

done
