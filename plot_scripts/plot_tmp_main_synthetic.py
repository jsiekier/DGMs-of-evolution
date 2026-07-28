import os
import time
import seaborn as sns
import pickle as pkl
import pandas as pd

from plot_tmp_2 import make_sum_paper_plot_no_wf, linkage_correlation_filter

if __name__ == '__main__':
    plot_path = 'plots/synthetic_new_models_gt_plus_noise'
    time_series_LD=False
    use_abs = False  # True
    exp_idx = 1
    work_idx = 0

    wf_file_end = 'no_n_new_N_s_pos.pkl'  # '_all_real_noise.pkl'#
    use_mean = False
    work_from = ['mogon', 'work_lap'][work_idx]  # # 'work_lab'#'#work_lab'#
    tmp_data_folder = ['/sync_data/', 'data/'][work_idx]
    num_submodels = [3, 1][work_idx]

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    plot_path += '/'

    finetuning = True
    use_only_test = True

    time_stamp = time.time()
    KDE_bandwidth = 0.05
    summary_method = [0, 1, 2][1]

    exp_0 = [ ]
    if work_idx == 0:
        noise_paths = ["0", "4", "8","12"]
        num_targets = ["10", "25", "50"] #
        sampling_noise = ['',"noise14"]
        coverage = [1000,40]
        Nsampling = [1000,100]
        use_gene =['no w', 'w']
        use_gene_str = ['0', '1']
        use_gene_end =['_z10_','_z10_']
        num_submodels = 3

        model_names_begin = ['VVAE_max_']
        model_names_end = ['_']
        plot_name = ['VAE']
        batch_size = [6]
        KLD = [4]  # ,3]#,3,3,3]


        input_data = []
        # Loop over parameters
        for noise_path in noise_paths:
            for i, (cov, sampling, sampling_str) in enumerate(zip(coverage, Nsampling, sampling_noise)):
                for num_target in num_targets:
                    animal = f"max_{noise_path}_r_t_{num_target}_n_{sampling_str}"
                    animal_no_noise = f"max_{noise_path}_r_t_{num_target}_n_"

                    for integrate_gene, integrate_gene_str, m_names_end in zip(use_gene, use_gene_str, use_gene_end):
                        nn_name = f"VVAE_max_{noise_path}_r_t_{num_target}_n{sampling_str}_usegene{integrate_gene_str}{m_names_end}KLD4"#5"
                        input_data.append([nn_name, integrate_gene, 6, True, animal, noise_path, animal_no_noise,
                                           '(' + str(sampling) + ',' + str(cov) + ')', num_target])

        '''
        ############################################################################################
        # hyperparameter search:
        noise_paths = ["0", "4", "8","12"]#
        num_targets = [ "25"] #
        sampling_noise = ["noise14"]  # ,"noise25"]# ["noise14"] #"",
        coverage = [40]  # ,50]# [40]#1000,
        Nsampling = [100]  # ,200]# [100]#1000,
        use_gene =[ 'w']  # ['w']#
        use_gene_str = ['1']  #['1']#
        use_gene_end = ['_']#['_', '_abs_']
        num_submodels = 3

        model_names_begin = ['VVAE_max_']  # ,'VVAE_max_']#,"miniVAE_max_","miniVAE_max_","miniVAE_max_"]
        model_names_end = ['_']  # ,'_sim_3_']#,'_','_sim_',]
        plot_name = ['VAE']  # ,'new']#,'1','2','3']
        batch_size = [6]  # ,5]#,5,5,5]
        KLD = [4, 5, 6]  # ,3]#,3,3,3]
        z_dims = [5, 10, 25]
        input_data = []
        # Loop over parameters
        for noise_path in noise_paths:
            for i, (cov, sampling, sampling_str) in enumerate(zip(coverage, Nsampling, sampling_noise)):
                animal = f"max_{noise_path}_r_t_25_n_{sampling_str}"
                animal_no_noise = f"max_{noise_path}_r_t_25_n_"
                for kld in KLD:
                    for z_dim in z_dims:

                    #for integrate_gene, integrate_gene_str, m_names_end in zip(use_gene, use_gene_str, use_gene_end):
                        #nn_name = f"VVAE_max_{noise_path}_r_t_{num_target}_n{sampling_str}_usegene{integrate_gene_str}{m_names_end}KLD5"
                        nn_name=f"VVAE_gene1_tbs6_n50_negsoft2_z{z_dim}_1enc3_{animal}_KLD{kld}"
                        input_data.append([nn_name, str(z_dim)+'-'+str(kld), 6, True, animal, noise_path, animal_no_noise,
                                           '(' + str(sampling) + ',' + str(cov) + ')', 25])

        ###########################################################################################
        '''
        sns.set_style("whitegrid")
        debugg = 1
        chrom = '2L'
        print('Third plot finished')
        ti = 0
        identity_gen = afc_gen = 5  # 3#
        max_time_bz = num_train_gens = 6  # 4#
        num_generations = 16
        cut_time = 1
        target_knowledge = True
        generations_separate = False#True
        if generations_separate:
            columns = ['dataset_name', 'snp type', 'submodel_num', 'result_metric', 'result_metric_name',
                       'dataset_type',
                       'p_{LD} influence', 'model name', 'Chrom', 'TestGen']
        else:
            columns = ['dataset_name', 'snp type', 'submodel_num', 'result_metric', 'result_metric_name',
                       'dataset_type',
                       'p_{LD} influence', 'model name', 'Chrom']

        df_result = pd.DataFrame(columns=columns)
        #'''
        df_result=make_sum_paper_plot_no_wf(tmp_data_folder, input_data, num_submodels, chrom='2L',max_time_bz=max_time_bz,
                                  num_generations=num_generations,cut_time = cut_time,wf_file_end=wf_file_end,work_from=work_from,
                                    df=df_result,columns=columns,target_knowledge=target_knowledge,
                                    generations_separate=generations_separate,identity_gen=identity_gen,ti=ti)
        pkl.dump(df_result, open(plot_path + 'simulation_results.pkl', 'wb'))
        print('saved the simulation performance')

        use_gene = ['w']  # ['w']#
        use_gene_str = ['1']  # ['1']#
        input_data = []
        # Loop over parameters
        for noise_path in noise_paths:
            for i, (cov, sampling, sampling_str) in enumerate(zip(coverage, Nsampling, sampling_noise)):
                for num_target in num_targets:
                    animal = f"max_{noise_path}_r_t_{num_target}_n_{sampling_str}"
                    animal_no_noise = f"max_{noise_path}_r_t_{num_target}_n_"

                    for integrate_gene, integrate_gene_str, m_names_end in zip(use_gene, use_gene_str, use_gene_end):
                        nn_name = f"VVAE_max_{noise_path}_r_t_{num_target}_n{sampling_str}_usegene{integrate_gene_str}{m_names_end}KLD4"#5"
                        input_data.append([nn_name, integrate_gene, 6, True, animal, noise_path, animal_no_noise,
                                           '(' + str(sampling) + ',' + str(cov) + ')', num_target])
        for linkage_def_idx in [3]:#2,3,0,1
            print(linkage_def_idx)
            model_rows, afc_rows, linkage_rows = [], [], []
            model_rows,afc_rows,linkage_rows=linkage_correlation_filter(model_rows, afc_rows, linkage_rows,input_data,linkage_def_idx,num_neigbors=50,
                                plot_path=plot_path+str(linkage_def_idx),
                                       num_submodels=num_submodels,chrom=chrom,debugg=debugg,use_abs=use_abs,
                                       tmp_data_folder=tmp_data_folder,afc_gen=5,num_train_gens=6,work_from=work_from,time_series_LD=time_series_LD) # for paper we had afc_gen=5?
            df_model, df_afc, df_link = pd.DataFrame(model_rows), pd.DataFrame(afc_rows), pd.DataFrame(linkage_rows)
            # save python dataframe :) and plot later :)
            pkl.dump([df_model, df_afc, df_link], open(plot_path +str(linkage_def_idx)+ 'LD_spearman_corr.pkl', 'wb'))
            print('saved all linkage correlations :)')
