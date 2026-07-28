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

### Additionally, to run LDx (+ alignment)
* BWA
* SAMtools
* BCFtools
* Pearl
* wgsim


##  Usage
### 1. Experiments on synthetic datasets:
#### 1.1. Create Artificial Haplotypes

```bash
python create_mimicree_files.py
```

#### 1.2. Run Mimicree2 Simulations

```bash
./slurm_scripts/1_batch_slurm_simulate.sh
```

#### 1.3. Save Created Paths

```bash
sbatch ./slurm_scripts/2_slurm_single_write_paths
```

#### 1.4. Preprocess Sync Files for Neural Network Training

```bash
./slurm_scripts/3_batch_slurm_preprocess.sh
```

#### 1.5. Estimate Ne with PoolSeq R Library

```bash
sbatch ./slurm_scripts/3a_slurm_single_estimateNe
```

#### 1.6. Estimate Selection Coefficients (s)

```bash
./slurm_scripts/3b_batch_slurm_estimate_s.sh
```

#### 1.7. Train VAE Model

```bash
./slurm_scripts/4_batch_slurm_model_training.sh
```

#### 1.8.1 Evaluate VAE + WF Model

```bash
./slurm_scripts/5_batch_slurm_model_evaluation.sh
```

#### 1.8.2 LDx: LD estimation 
```bash
./LDx/batch_slurm_LDx.sh
```

#### 1.9. Generate Plots

* Short Data Analysis Plot for Appendix:

```bash
python plot_scripts/main_data_analysis_exp_II.py
```

* Evaluation Plots:

```bash
python plot_scripts/plot_tmp_main_synthetic.py
python plot_scripts/plot_linkage_correlations.py
```

### 2. Experiments on empirical dataset:

#### 2.1 Download empirical data:
processed sync file from https://doi.org/10.5061/dryad.rr137kn, 
the raw reads (for LDx) from the European Nucleotide Archive under project PRJEB29281,
M252 Madagascar reference genome from https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_000820565.1/
and the phased haplotypes of the founder population from Dryad (doi:10.5061/dryad.744p394)

#### 2.1.1. Cut Sync Files to drup unrelevant information/ cut according to chromosome

```bash
python sync_cut.py --input_sync "Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID.sync" --cut_sync "Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID_cut.sync" --output_prefix "Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID_"
```

#### 2.2 manually set all paths in variables.py
```bash
fix_variables = {'home': {'experimental':
 'estimated_s': [   [   '<path_to_s_estimation>/s_experimental.txt']],
         'sync': '<path_to_sync>/Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID_cut.sync',
         'sync_process': '<path_to_sync_process>/Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID.pkl',
         'eval_positions':'<path_to_eval_idx>/experimental_pos_LD.pkl',
         'start_haplos': '<path_to_haplos>/haplos',
         'LDx':{'2L':[ldx_path+'2.txt',ldx_path+'4.txt',ldx_path+'6.txt',ldx_path+'7.txt',ldx_path+'9.txt'],
                '2R':[ldx_path+'2_2R.txt',ldx_path+'4_2R.txt',ldx_path+'6_2R.txt',ldx_path+'7_2R.txt',ldx_path+'9_2R.txt'],
                '3L':[ldx_path+'2_3L.txt',ldx_path+'4_3L.txt',ldx_path+'6_3L.txt',ldx_path+'7_3L.txt',ldx_path+'9_3L.txt'],
                '3R':[ldx_path+'2_3R.txt',ldx_path+'4_3R.txt',ldx_path+'6_3R.txt',ldx_path+'7_3R.txt',ldx_path+'9_3R.txt'],
                 'X':[ldx_path+'2_X.txt',ldx_path+'4_X.txt',ldx_path+'6_X.txt',ldx_path+'7_X.txt',ldx_path+'9_X.txt']}
}}
```

#### 2.2. Preprocess Sync Files for Neural Network Training

```bash
python sync_cut.py
python preprocessing -animal 'experimental'

```

#### 2.3.1 estimate Ne + s + WF
```bash
sbatch 3a_slurm_single_estimateNe_experimental
sbatch 3c_slurm_single_declare_eval_positions "experimental"
sbatch 5_slurm_single_WF_execution "experimental"
```
#### 2.3.2 LDx: LD estimation 
```bash
./LDx/batch_slurm_LDx.sh
```


#### 2.4 execute VAE and evaluate
```bash
sbatch 4_slurm_single_VAE_train "experimental" "VAE_max_usegene1_n100" 1 4 4 0.0 0.0 10 50 'None' 'tanh' 1 2 0 0 'sim'
sbatch 5_slurm_single_VAE_eval "experimental" "VAE_max_usegene1_n100" 1 5

```

#### 2.5. Generate Plots

* Short Data Analysis Plot for Appendix:

```bash
python plot_scripts/main_experimental_data_analysis.py
```

* Evaluation Plots:

```bash
python plot_scripts/plot_tmp_experimental.py
python plot_scripts/plot_linkage_correlations.py
```
## Contact

For any questions, please contact siekiera@uni-mainz.de.

