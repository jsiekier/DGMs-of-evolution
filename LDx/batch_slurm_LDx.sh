#!/bin/bash

# Define parameter lists
noise_paths=("2" "4" "6" "7" "9")
chroms=("2L" "2R" "3R" "3L" "X")

# Loop over parameters and submit SLURM jobs
for noise_path in "${noise_paths[@]}"; do
    for chrom in "${chroms[@]}"; do


        haplo_file="/lustre/project/ki-dm/siekiera/simulation_data/"
        out_file="/lustre/project/ki-dm/siekiera/LDx/vcf/${noise_path}_${chrom}variants.filtered.vcf.gz"
        # Submit SLURM job
        out_file1="/lustre/project/ki-dm/siekiera/LDx/fastq/Base_${noise_path}/Base_${noise_path}_R1.fastq.gz"
        out_file2="/lustre/project/ki-dm/siekiera/LDx/fastq/Base_${noise_path}/Base_${noise_path}_R2.fastq.gz"

        sim_dedup="/lustre/project/ki-dm/siekiera/LDx/bam_files/${noise_path}mapped.dedup.bam"
        sim_sorted="/lustre/project/ki-dm/siekiera/LDx/bam_files/${noise_path}sim.sorted.bam"
        sim_sam="/lustre/project/ki-dm/siekiera/LDx/sam/${noise_path}_${chrom}.dedup.sorted.sam"
        sim_sync="/lustre/project/ki-dm/siekiera/gen0_fasta/gen0_max${noise_path}sim.sync"
        out_file_sampled="/lustre/project/ki-dm/siekiera/gen0_fasta/gen0_max_haplos_sampled${noise_path}"
        vcf="/lustre/project/ki-dm/siekiera/LDx/vcf/${noise_path}_${chrom}variants.vcf.gz"
        ldx_out="/lustre/project/ki-dm/siekiera/LDx/ldx_out_new/${noise_path}_${chrom}.txt"
        tmp_bam="/lustre/project/ki-dm/siekiera/LDx/bam_files/${noise_path}tmp.bam"
        tmp_fixmate="/lustre/project/ki-dm/siekiera/gen0_fasta/gen0_max_haplos_sampled${noise_path}.fixmate.bam"
        out_file1_trim="/lustre/project/ki-dm/siekiera/LDx/fast_q/trim_ERR${noise_path}_1.fastq.gz"
        out_file2_trim="/lustre/project/ki-dm/siekiera/LDx/fast_q/trim_ERR${noise_path}_2.fastq.gz"
        sbatch slurm_LDx "$haplo_file" "$out_file" "$out_file1" "$out_file2" "$sim_dedup" "$sim_sorted" "$sim_sam" "$sim_sync" "$out_file_sampled" "$vcf" "$ldx_out" "$tmp_bam" "$tmp_fixmate" "$out_file1_trim" "$out_file2_trim" "$chrom"
    done
done

