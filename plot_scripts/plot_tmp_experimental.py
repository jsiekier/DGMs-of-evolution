import os
import time
import pickle as pkl
import pandas as pd

from plot_tmp_2 import linkage_correlation_filter,make_sum_paper_plot_no_wf


if __name__ == '__main__':
    plot_path = '/plots/exp_new_models_other_testset'#exp_haplo_no_noise' #exp_flip_wf_noise' #exp_rec' # #recombi_tbs6_LDoverTime_untrained'#recombiexp' #experimental_KLD_z_experiment_noarcsin' #'_base_fSF_sub1'#newnets_uncomplete'  # _sub1 #noise_sub1'#
    use_abs = False  # True
    use_LD_over_time=False#True#False#True
    #exp_idx = 1
    work_idx =0

    wf_file_end ='no_n_new_N_s_pos.pkl'  # '_all_real_noise.pkl'#
    use_mean = False
    work_from = ['mogon', 'work_lap'][work_idx]  # # 'work_lab'#'#work_lab'#
    tmp_data_folder = ['/sync_data/', 'data/'][work_idx]
    num_submodels = [3, 1][work_idx]

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    plot_path += '/'
    finetuning = True
    use_only_test = True
    # show_random = 2  # 0= no random 1=optimistic random 2=pessimistic random
    # '''
    time_stamp = time.time()
    KDE_bandwidth = 0.05
    summary_method = [0, 1, 2][1]

    exp_0 = [
        ('VVAE_gene1_tbs4_n100_negsoft2_z10_1enc3_experimental_flip_KLD4', 'w', 4, True, 'experimental_flip', "4",'experimental_flip', '', 'w'),
        ('VVAE_gene0_tbs4_n100_negsoft2_z10_1enc3_experimental_flip_KLD4', 'no w', 4, True, 'experimental_flip', "4",'experimental_flip', '', 'no w'),
    ]

    ti=0
    max_time_bz = [4,3,2][ti]
    num_generations = [7,7,7][ti]
    cut_time = [1,1,1][ti]
    identity_gen = [3,2,1][ti]
    afc_gen = [4,3,2][ti]
    num_train_gens =[4,3,2][ti]
    target_knowledge=False

    chroms=['2L','2R','3L','3R','4','X']
    generations_separate = True
    if generations_separate:
        columns = ['dataset_name', 'snp type', 'submodel_num', 'result_metric', 'result_metric_name', 'dataset_type',
                   'p_{LD} influence', 'model name', 'Chrom','TestGen']
    else:
        columns = ['dataset_name', 'snp type', 'submodel_num', 'result_metric', 'result_metric_name', 'dataset_type',
                   'p_{LD} influence', 'model name', 'Chrom']


    df_result = pd.DataFrame(columns=columns)


    for chrom in chroms:

        df_result=make_sum_paper_plot_no_wf(tmp_data_folder, exp_0, num_submodels, chrom=chrom,max_time_bz=max_time_bz,
                                  num_generations=num_generations,cut_time = cut_time,wf_file_end=wf_file_end,work_from=work_from,
                                            df=df_result,columns=columns,target_knowledge=target_knowledge,
                                            generations_separate=generations_separate,identity_gen=identity_gen,ti=ti)
        # save the df:
        pkl.dump(df_result, open(plot_path + 'simulation_results.pkl', 'wb'))
        print('saved the simulation performance')



    debugg=True
    chroms = ['2L', '2R', '3L', '3R', 'X']
    exp_0=[('VVAE_gene1_tbs4_n100_negsoft2_z10_1enc3_experimental_flip_KLD4', 'VAE', 4, True, 'experimental_flip', "4",'experimental_flip', '', 'VAE')]
    for linkage_def_idx in [1,3]:
        model_rows, afc_rows, linkage_rows=[],[],[]
        for chrom in chroms:
            print(linkage_def_idx)
            model_rows,afc_rows,linkage_rows= linkage_correlation_filter(model_rows, afc_rows, linkage_rows,exp_0,linkage_def_idx,num_neighbors=50,
                                plot_path=plot_path+str(linkage_def_idx),num_submodels=num_submodels,chrom=chrom,
                                       debugg=debugg,use_abs=use_abs,tmp_data_folder=tmp_data_folder,afc_gen=afc_gen,
                                                                         num_train_gens=num_train_gens,work_from=work_from,time_series_LD=use_LD_over_time)

            df_model, df_afc, df_link = pd.DataFrame(model_rows), pd.DataFrame(afc_rows), pd.DataFrame(linkage_rows)
            # save python dataframe :) and plot later :)
            pkl.dump([df_model, df_afc, df_link], open(plot_path +str(linkage_def_idx)+ 'LD_spearman_corr.pkl', 'wb'))
            print('saved all linkage correlations :)')
