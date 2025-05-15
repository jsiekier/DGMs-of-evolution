# DGMs-of-Evolution

This repository provides the code for the paper **Deep Generative Models of Evolution: SNP-level Population Adaptation by Genomic Linkage Incorporation**, submitted to 24th International Workshop on Data Mining in Bioinformatics (BioKDD25). 
The code is designed for creating artificial haplotypes, running simulations, preprocessing data, and training a VAE model for evolutionary data analysis.

## Getting Started

### Prerequisites

* Python, Java, R
* Slurm for batch processing
* Mimicree2 (mim2-v206.jar)
* PoolSeq R library
* Python dependencies can be installed via:

```bash
pip install -r requirements.txt
```



##  Usage

### 1. Create Artificial Haplotypes

```bash
python create_mimicree_files.py
```

### 2. Run Mimicree2 Simulations

```bash
sbatch ./slurm_scripts/1_batch_slurm_simulate.sh
```

### 3. Save Created Paths

```bash
sbatch ./slurm_scripts/2_slurm_single_write_paths
```

### 4. Preprocess Sync Files for Neural Network Training

```bash
sbatch ./slurm_scripts/3_batch_slurm_preprocess.sh
```

### 5. Estimate Ne with PoolSeq R Library

```bash
sbatch ./slurm_scripts/3a_slurm_single_estimateNe
```

### 6. Estimate Selection Coefficients (s)

```bash
sbatch ./slurm_scripts/3b_batch_slurm_estimate_s.sh
```

### 7. Train VAE Model

```bash
sbatch ./slurm_scripts/4_batch_slurm_model_training.sh
```

### 8. Evaluate VAE + WF Model

```bash
sbatch ./slurm_scripts/5_batch_slurm_model_evaluation.sh
```

### 9. Generate Plots

* Short Data Analysis Plot for Appendix:

```bash
python plot_scripts/main_data_analysis_exp_II.py
```

* Evaluation Plots:

```bash
python plot_scripts/plot_tmp_2.py
python plot_scripts/plot_linkage_correlations.py
```



## Contact

For any questions, please contact siekiera@uni-mainz.de.

