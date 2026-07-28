import pickle as pkl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from variables import fix_variables
from scipy import stats

import tensorflow as tf
import matplotlib
from scipy.optimize import minimize
from joblib import cpu_count
from plot_scripts.evaluation_metrics import all_metrics, all_metrics_list
# from pyemd import emd
palette = sns.color_palette("viridis", as_cmap=True)

red = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
other_colors = [(0.0, 0.0, 0.0),
                (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                (1.0, 0.4980392156862745, 0.054901960784313725),
                (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
x_sign = 'X'

markers = ['o'] * 10

plot_type = '.pdf'






def load_model_data_noise_less_info(entry,summary_method=1,chrom='2L',work_from='mogon',num_submodels=3,tmp_data_folder='',target_knowledge=True,identity_gen=5):
    #model_name, plot_name, time_bz, _, animal_noise, ds_name, animal=entry
    (model_name, plot_name, time_bz, finetuning, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
    eval_path = fix_variables[work_from][animal]['eval_positions']
    print(eval_path)
    linkage_data = pkl.load(open(eval_path, 'rb'))
    if target_knowledge:
        num_targets=int(len(linkage_data['selected'])/100)
        print('--------num targets VAE',num_targets)
        real_selected_points = [linkage_data['selected'][50 + (s_idx * 100)] for s_idx in range(num_targets)]

        idx_lists = [linkage_data['unselected'], real_selected_points]#[[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data=[list(range(len(linkage_data['selected']),len(linkage_data['selected'])+len(linkage_data['unselected']))),
                             [list(range(len(linkage_data['selected'])))[50 + (s_idx * 100)] for s_idx in range(num_targets)]]
    else:
        idx_lists = [linkage_data['unselected'][chrom], linkage_data['selected'][chrom]]  # [[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data = [
            list(range(len(linkage_data['selected'][chrom]), len(linkage_data['selected'][chrom]) + len(linkage_data['unselected'][chrom]))),
            list(range(len(linkage_data['selected'][chrom])))]


    noise_freqs = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        noise_freqs = np.squeeze(noise_freqs[chrom])
    print(work_from,animal_noise,chrom,noise_freqs.shape,flush=True)
    noise_freqs = np.transpose(noise_freqs, (2, 0, 1))[1:]
    gt_freqs = pkl.load(open(fix_variables[work_from][animal]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        gt_freqs = np.squeeze(gt_freqs[chrom])
    gt_freq = np.transpose(gt_freqs, (2, 0, 1))[1:]

    mean_gt = np.mean(gt_freq, axis=2)
    std_gt = np.std(gt_freq, axis=2)

    #####################################################################

    # remember we start idx with gen 1
    noise_freqs = noise_freqs[identity_gen]



    plot_data = []
    for submodel_num in range(num_submodels):
        model_name_ = model_name + '_' + str(submodel_num)
        if finetuning:
            model_name_ += '_fine'
        path = tmp_data_folder + 'models/' + model_name_ + '/eval_data/train_data.pkl'
        in_stream = open(path, 'rb')
        print(path)
        data = pkl.load(in_stream)
        prediction = data['freqs']
        if chrom!='':
            prediction=prediction[chrom]
        print(prediction.shape)


        std_pred = np.std(prediction, axis=1)
        mean_pred = np.mean(prediction, axis=1)

        #subtracted_mean = np.abs(mean_gt[time_bz - 1:] - mean_pred)
        #substracted_std = np.abs(std_gt[time_bz - 1:] - std_pred)

        #print('DEBUG shapes of MODEL prediction:', prediction.shape, 'gt freq shape', gt_freq.shape,std_gt.shape,std_pred.shape,time_bz)
        prediction = np.transpose(prediction, (0, 2, 1))
        gt_freq_replicated = np.repeat(np.expand_dims(gt_freq, axis=2), repeats=10, axis=2)
        gt_freq_replicated = np.reshape(gt_freq_replicated,
                                        (gt_freq_replicated.shape[0], gt_freq_replicated.shape[1],
                                         gt_freq_replicated.shape[2] * gt_freq_replicated.shape[3]))

        remaining_results = {'std': [np.mean(np.abs(std_gt[time_bz - 1:,idx_lists[0]] - std_pred[:,idx_list_model_data[0]]), axis=1),
                                     np.mean(np.abs(std_gt[time_bz - 1:,idx_lists[1]] - std_pred[:,idx_list_model_data[1]]), axis=1)],

                             'mean':  [np.mean(np.abs(mean_gt[time_bz - 1:,idx_lists[0]] - mean_pred[:,idx_list_model_data[0]]), axis=1),
                                     np.mean(np.abs(mean_gt[time_bz - 1:,idx_lists[1]] - mean_pred[:,idx_list_model_data[1]]), axis=1)],

                            'MAE':[np.mean(np.abs(gt_freq_replicated[time_bz - 1:, idx_lists[0]] - prediction[:, idx_list_model_data[0]]), axis=(1,2)),# gt_freq (generation?,snps,reps)
                                   np.mean(np.abs(gt_freq_replicated[time_bz - 1:, idx_lists[1]] - prediction[:, idx_list_model_data[1]]), axis=(1,2))],
                            'RMSE':[np.mean(np.abs(gt_freq_replicated[time_bz - 1:, idx_lists[0]] - prediction[:, idx_list_model_data[0]])**2, axis=(1,2)),# gt_freq (generation?,snps,reps)
                                   np.mean(np.abs(gt_freq_replicated[time_bz - 1:, idx_lists[1]] - prediction[:, idx_list_model_data[1]])**2, axis=(1,2))]

                             }
        for metric in all_metrics_list:
            remaining_results[metric.internal_name] = []





        ##############
        #print('DEBUG shapes of prediction:', prediction.shape, 'gt freq shape', gt_freq.shape)
        debug_selected_mean = np.mean(prediction[:, idx_list_model_data[1][:2]], axis=-1)
        #print('example points VAE')
        #for i_d in range(2):
        #    for gen in range(debug_selected_mean.shape[0]):
        #        print(np.round(debug_selected_mean[gen, i_d], 2), end='\t')
        #    print()
        #############



        for idx_list_gt,idx_list_model in zip(idx_lists,idx_list_model_data):
            for metric in all_metrics_list:
                remaining_results[metric.internal_name].append(np.zeros((prediction.shape[0])))
            for gen in range(prediction.shape[0]):
                for metric in all_metrics[3]:
                    #if metric.internal_name == 'relative_mean_error':
                        #print('VAE:',gen)
                        #print('VAE',
                        #      gt_freq[gen, idx_list_gt].shape, 'pred',
                        #      prediction[gen, idx_list_model].shape, 'noise',
                        #      noise_freqs[idx_list_gt].shape)
                    remaining_results[metric.internal_name][-1][gen] = metric.eval_function(
                        gt_freq[gen + (time_bz - 1), idx_list_gt], prediction[gen, idx_list_model], noise_freqs[idx_list_gt],
                        summary_method)
        in_stream.close()
        plot_data.append(remaining_results)
    return plot_data
def read_wf_file_noise_less_info(animal_noise,animal,summary_method=1,chrom='2L',tmp_data_folder='',wf_file_end='',work_from='',target_knowledge=True,identity_gen=5):
    num_train_it=0
    animal_noise_=animal_noise.replace('_flip','')
    wf_file_name = tmp_data_folder + 'wf_data/wf_simulations_' + animal_noise_ + '_' + str(num_train_it) + wf_file_end
    print('WF file name',wf_file_name)
    in_stream = open(wf_file_name, 'rb')
    wf_data = pkl.load(in_stream)
    in_stream.close()
    gt_data = dict()
    ###################################
    #was data flipped? then flipp also WF results to match the dataset:
    flips=[]
    if 'flip' in animal_noise:
        # read flip positions
        sync_file = fix_variables[work_from][animal]["sync_process"]
        data = pkl.load(open(sync_file, 'rb'))
        print(animal,data.keys(),flush=True)
        if 'flips' in data:
            flips = data['flips'][chrom]
        else:
            eval_path = fix_variables[work_from][animal]['eval_positions']
            idx_list_dict = pkl.load(open(eval_path, 'rb'))
            idx_list = np.asarray(idx_list_dict['selected'][chrom]).tolist() + np.asarray(idx_list_dict['unselected'][chrom]).tolist()
            nucleotide_data_flip = data['nucleotide_idx'][chrom]
            sync_file = fix_variables[work_from][animal_noise_]["sync_process"]
            data = pkl.load(open(sync_file, 'rb'))
            nucleotide_data_org=data['nucleotide_idx'][chrom]
            flips=[]
            for i_dx,i in enumerate(idx_list):
                if nucleotide_data_org[i][0]!=nucleotide_data_flip[i][0]:
                    flips.append(i_dx)
            print('FLIP analysis',nucleotide_data_org.shape[0],len(flips),flush=True)



    ####################################################################################################
    eval_path = fix_variables[work_from][animal]['eval_positions']
    print(eval_path)
    linkage_data = pkl.load(open(eval_path, 'rb'))
    if target_knowledge:
        num_targets=int(len(linkage_data['selected'])/100)
        print('--------num targets',num_targets)
        real_selected_points = [linkage_data['selected'][50 + (s_idx * 100)] for s_idx in range(num_targets)]
        idx_lists = [linkage_data['unselected'], real_selected_points]#[[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data=[list(range(len(linkage_data['selected']),len(linkage_data['selected'])+len(linkage_data['unselected']))),
                             [list(range(len(linkage_data['selected'])))[50 + (s_idx * 100)] for s_idx in range(num_targets)]]
        print('selected positions',real_selected_points)
    else:
        idx_lists = [linkage_data['unselected'][chrom], linkage_data['selected'][chrom]]#[[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data=[list(range(len(linkage_data['selected'][chrom]),len(linkage_data['selected'][chrom])+len(linkage_data['unselected'][chrom]))),
                             list(range(len(linkage_data['selected'][chrom])))]
        print(len(idx_lists[0]),len(idx_lists[1]))


    ####################################################################################################
    noise_freqs = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        noise_freqs = np.squeeze(noise_freqs[chrom])
    noise_freqs = np.transpose(noise_freqs, (2, 0, 1))[1:]
    gt_stuff=pkl.load(open(fix_variables[work_from][animal]['sync_process'], 'rb'))
    gt_freqs = gt_stuff['alleles']
    position_data=gt_stuff['distances']

    if chrom != '':
        gt_freqs = np.squeeze(gt_freqs[chrom])
        position_data=np.squeeze(position_data[chrom])
    #print('selected positions', position_data[real_selected_points])
    gt_freq = np.transpose(gt_freqs, (2, 0, 1))[1:]

    mean_gt = np.mean(gt_freq, axis=2)
    std_gt = np.std(gt_freq, axis=2)

    #####################################################################

    # remember we start idx with gen 1
    noise_freqs = noise_freqs[identity_gen]
    ####################################################################################################


    #gt_data['mean'] = np.mean(gt_freq, axis=2)
    #gt_data['std'] =np.std(gt_freq, axis=2)
    prediction = wf_data['freqs'][chrom]
    # integrate flips is data was flipped before to sync wf data and flipped preprocessing

    print(prediction.shape,flush=True)
    prediction[:,:,:, flips] = 1 - prediction[:,:,:, flips]




    std_pred = np.std(prediction, axis=(0, 2))[1:]
    #print('DEBUG shapes WF:', prediction.shape, 'gt freq shape', gt_freq.shape,std_pred.shape,gt_data['std'].shape)

    #all_stds = np.abs(std_pred - gt_data['std'])

    mean_pred = np.mean(prediction, axis=(0, 2))[1:]
    #all_abs_means_ = np.abs(gt_data['mean'] - mean_pred)
    #print('DEBUG shapes of prediction:', prediction.shape, 'gt freq shape', gt_freq.shape)
    prediction = np.transpose(prediction, (1, 3, 0, 2))
    prediction = np.reshape(prediction,(prediction.shape[0], prediction.shape[1], prediction.shape[2] * prediction.shape[3]))
    #TODO look if rashape is right

    print('DEBUG shapes of prediction:', prediction.shape, 'gt freq shape',
          gt_freq.shape,'std_gt.shape',std_gt.shape,'std_pred.shape',std_pred.shape)



    gt_freq_replicated=np.repeat(np.expand_dims(gt_freq,axis=-1),repeats=10,axis=-1)
    gt_freq_replicated=np.reshape(gt_freq_replicated,
                                    (gt_freq_replicated.shape[0],gt_freq_replicated.shape[1],gt_freq_replicated.shape[2]*gt_freq_replicated.shape[3]))

    prediction = prediction[1:]




    debug_selected_mean=np.mean(prediction[:,idx_list_model_data[1][:2]],axis=-1)
    print('example points')
    for i_d in range(2):
        for gen in range(debug_selected_mean.shape[0]):
            print(np.round(debug_selected_mean[gen,i_d],2),end='\t')
        print()

    remaining_results = {
        'std': [np.mean(np.abs(std_gt[:, idx_lists[0]] - std_pred[:, idx_list_model_data[0]]), axis=1),
                np.mean(np.abs(std_gt[:, idx_lists[1]] - std_pred[:, idx_list_model_data[1]]), axis=1)],

        'mean': [np.mean(np.abs(mean_gt[:, idx_lists[0]] - mean_pred[:, idx_list_model_data[0]]), axis=1),
                 np.mean(np.abs(mean_gt[:, idx_lists[1]] - mean_pred[:, idx_list_model_data[1]]), axis=1)],

        'MAE':[np.mean(np.abs(gt_freq_replicated[:, idx_lists[0]] - prediction[:, idx_list_model_data[0]]), axis=(1,2)),# gt_freq (generation?,snps,reps)
               np.mean(np.abs(gt_freq_replicated[:, idx_lists[1]] - prediction[:, idx_list_model_data[1]]), axis=(1,2))],
        'RMSE':[np.mean(np.abs(gt_freq_replicated[:, idx_lists[0]] - prediction[:, idx_list_model_data[0]])**2, axis=(1,2)),# gt_freq (generation?,snps,reps)
               np.mean(np.abs(gt_freq_replicated[:, idx_lists[1]] - prediction[:, idx_list_model_data[1]])**2, axis=(1,2))]

    }

    noise_freqs_repeated=np.repeat(np.expand_dims(noise_freqs,axis=0),axis=0,repeats=gt_freq.shape[0])
    noise_freqs_repeated_std=np.std(noise_freqs_repeated,axis=2)
    noise_freqs_repeated_mean=np.mean(noise_freqs_repeated,axis=2)

    identity_result= {
        'std': [np.mean(np.abs(std_gt[:, idx_lists[0]] - noise_freqs_repeated_std[:, idx_lists[0]]), axis=1),
                np.mean(np.abs(std_gt[:, idx_lists[1]] - noise_freqs_repeated_std[:, idx_lists[1]]), axis=1)],

        'mean': [np.mean(np.abs(mean_gt[:, idx_lists[0]] - noise_freqs_repeated_mean[:, idx_lists[0]]), axis=1),
                 np.mean(np.abs(mean_gt[:, idx_lists[1]] - noise_freqs_repeated_mean[:, idx_lists[1]]), axis=1)],

        'MAE':[np.mean(np.abs(gt_freq[:, idx_lists[0]] - noise_freqs_repeated[:, idx_lists[0]]), axis=(1,2)),# gt_freq (generation?,snps,reps)
               np.mean(np.abs(gt_freq[:, idx_lists[1]] - noise_freqs_repeated[:, idx_lists[1]]), axis=(1,2))],
        'RMSE':[np.mean(np.abs(gt_freq[:, idx_lists[0]] - noise_freqs_repeated[:, idx_lists[0]])**2, axis=(1,2)),# gt_freq (generation?,snps,reps)
               np.mean(np.abs(gt_freq[:, idx_lists[1]] - noise_freqs_repeated[:, idx_lists[1]])**2, axis=(1,2))]

    }



    #,TODO 'MAE','RMSE'

    for metric in all_metrics_list:
        remaining_results[metric.internal_name] = []

    for idx_list_gt, idx_list_model in zip(idx_lists, idx_list_model_data):
        for metric in all_metrics_list:
            remaining_results[metric.internal_name].append(np.zeros((prediction.shape[0])))

        for gen in range(prediction.shape[0]):
            for metric in all_metrics[3]:

                remaining_results[metric.internal_name][-1][gen] = metric.eval_function(gt_freq[gen, idx_list_gt],
                                                                                        prediction[gen, idx_list_model],
                                                                                        noise_freqs[idx_list_gt],
                                                                                        summary_method)


        print()

    return remaining_results,identity_result
def make_sum_paper_plot_no_wf(tmp_data_folder, input_data, num_submodels,
                              chrom='2L',max_time_bz=6,num_generations=16,
                              cut_time = 1,wf_file_end='',work_from='mogon',df=None,columns=[],
                              target_knowledge=True,generations_separate=False,identity_gen=5,ti=0):

    all_noise_ds=dict()
    for entry in input_data:
        (model_name, plot_name, time_bz, fine_tune, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
        all_noise_ds[animal_noise+'\t'+animal]=[ds_name,dataset_type,p_LD_influence]
    if wf_file_end!='':
        # if wf data exists:
        for key,values in all_noise_ds.items():
            animal_noise,animal=key.split('\t')
            ds_name, dataset_type, p_LD_influence=values
            # now calculate wf stuff :)
            print([ds_name,dataset_type,p_LD_influence])
            plot_data,identity_results=read_wf_file_noise_less_info(animal_noise,animal,summary_method=1,chrom=chrom,
                                                   tmp_data_folder=tmp_data_folder,wf_file_end=wf_file_end,
                                                   work_from=work_from,target_knowledge=target_knowledge,identity_gen=identity_gen)
            for metric_name in ['relative_mean_error', 'relative_std_error','mean','std','MAE','RMSE']:
                for snp_idx, snp_type in enumerate([ 'unselected','selected']):
                    if metric_name in ['MAE'] and snp_idx==1:
                        print('WF',np.mean(plot_data[metric_name][snp_idx][max_time_bz+cut_time-1:num_generations], axis=0),
                              plot_data[metric_name][snp_idx][max_time_bz+cut_time-1: num_generations])
                        print('id',np.mean(identity_results[metric_name][snp_idx][max_time_bz+cut_time-1:num_generations], axis=0),
                              identity_results[metric_name][snp_idx][max_time_bz+cut_time-1: num_generations])

                    if generations_separate:
                        test_gens_result=plot_data[metric_name][snp_idx][max_time_bz+cut_time-1:num_generations]
                        for test_gen, entry in enumerate(test_gens_result):
                            meta_data = (ds_name, snp_type, 0, entry, metric_name, dataset_type, p_LD_influence, 'WF model',chrom,test_gen)
                            row = {columns[i]: meta_data[i] for i in range(len(columns))}
                            # Append the dictionary as a new row to the DataFrame
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        # save results of identity baseline:
                        if metric_name in ['mean', 'std', 'MAE', 'RMSE']:
                            test_gens_result = identity_results[metric_name][snp_idx][max_time_bz + cut_time - 1:num_generations]
                            print(test_gens_result)
                            print(test_gens_result.shape)
                            for test_gen, entry_val in enumerate(test_gens_result):
                                meta_data = (
                                ds_name, snp_type, 0, entry_val, metric_name, dataset_type, p_LD_influence, 'ID',
                                chrom, test_gen)
                                row = {columns[i]: meta_data[i] for i in range(len(columns))}
                                # Append the dictionary as a new row to the DataFrame
                                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

                    else:
                        result_data_gens = np.mean(
                            plot_data[metric_name][snp_idx][max_time_bz + cut_time - 1:num_generations], axis=0)
                        meta_data = (ds_name, snp_type, 0, result_data_gens, metric_name, dataset_type, p_LD_influence, 'WF model',chrom)
                        row = {columns[i]: meta_data[i] for i in range(len(columns))}
                        # Append the dictionary as a new row to the DataFrame
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

                        # save results of identity baseline:
                        if metric_name in ['mean','std','MAE','RMSE']:
                            result_data_gens = np.mean(
                                identity_results[metric_name][snp_idx][max_time_bz + cut_time - 1:num_generations], axis=0)
                            meta_data = (ds_name, snp_type, 0, result_data_gens, metric_name, dataset_type, p_LD_influence, 'ID',chrom)
                            row = {columns[i]: meta_data[i] for i in range(len(columns))}
                            # Append the dictionary as a new row to the DataFrame
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    #TODO for ,'MAE','RMSE' baseline




    for ax_i, entry in enumerate(input_data):

        print(ax_i, 'load model',entry, flush=True)
        plot_data = load_model_data_noise_less_info(entry,  summary_method=1,chrom=chrom,
                                                    work_from=work_from,num_submodels=num_submodels,
                                                    tmp_data_folder=tmp_data_folder,target_knowledge=target_knowledge,identity_gen=identity_gen)
        #plot_data=np.asarray(plot_data)
        #plot_data_ds.append(plot_data)
        #(model_name, plot_name, time_bz, finetuning, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
        tbs=entry[2]
        train_gens=fix_variables[entry[4]]['num_train_generations'][ti]
        cut_time=train_gens-tbs
        print(entry[-2],end='\t')
        for submodel_num in range(num_submodels):
            submodel_data=plot_data[submodel_num]
            for metric_name in ['relative_mean_error','relative_std_error','mean','std','MAE','RMSE']:
                for snp_idx, snp_type in enumerate(['unselected','selected']):
                    if metric_name == 'relative_mean_error' and snp_idx == 1:
                        print('VAE',np.mean(submodel_data[metric_name][snp_idx][cut_time:num_generations],axis=0),
                              metric_name,submodel_data[metric_name][snp_idx][cut_time:num_generations]
                              )

                    if generations_separate:
                        test_gens_result=submodel_data[metric_name][snp_idx][cut_time:num_generations]
                        print(test_gens_result)
                        print(test_gens_result.shape)
                        for test_gen, entry_val in enumerate(test_gens_result):
                            meta_data = (entry[-4], snp_type, submodel_num, entry_val, metric_name, entry[-2],entry[-1],entry[1],chrom,test_gen)
                            row = {columns[i]: meta_data[i] for i in range(len(columns))}
                            # Append the dictionary as a new row to the DataFrame
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    else:
                        result_data_gens=np.mean(submodel_data[metric_name][snp_idx][cut_time:num_generations],axis=0)
                        meta_data=(entry[-4],snp_type,submodel_num,result_data_gens,metric_name,entry[-2],entry[-1],entry[1],chrom)
                        row = {columns[i]: meta_data[i] for i in range(len(columns))}
                        # Append the dictionary as a new row to the DataFrame
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df





def calculate_naive_sim_scores(gt_freqs, idx_list_pkl_data,use_abs=True,num_train_gens=6):
    gt_freqs=np.transpose(gt_freqs,(1,0,2))#(2,0,1))
    print(len(idx_list_pkl_data))
    print(gt_freqs.shape,flush=True)
    naive_sim_scores=[]

    valid_indices = []
    neighbors = []
    valid_indices_2 = []
    # this calculates the afc and probably makes the baseline too good
    #gt_freqs = gt_freqs - gt_freqs[:, 0:1]

    for i,idx in enumerate(idx_list_pkl_data):
        if idx < 50 or idx + 51 > gt_freqs.shape[0]:
            continue
        neig = np.concatenate([
            gt_freqs[idx - 50:idx, 0:num_train_gens],
            gt_freqs[idx + 1:idx + 51, 0:num_train_gens]
        ], axis=0)
        neighbors.append(neig)
        valid_indices.append(idx)
        valid_indices_2.append(i)

    neighbors = np.asarray(neighbors)
    focal_snp = gt_freqs[valid_indices, 0:num_train_gens]


    print('neighbors.shape',neighbors.shape,neighbors[0])
    for replicate in range(10):
        focal_SNP_rep=np.expand_dims(focal_snp[:,:,replicate],axis=1) # shape (1000,1,6)
        neighbors_rep=neighbors[:,:,:,replicate]# shape (1000,50,6)

        focus_enc2_normalized = tf.nn.l2_normalize(focal_SNP_rep, axis=-1)  # Shape: (batch_size, 1, latent_dim)
        neighbor_enc2_normalized = tf.nn.l2_normalize(neighbors_rep, axis=-1)  # Shape: (batch_size, n, latent_dim)

        similarity_scores = tf.matmul(focus_enc2_normalized, neighbor_enc2_normalized, transpose_b=True).numpy()
        similarity_scores=np.squeeze(similarity_scores)
        print(similarity_scores.shape,'shape')
        flatten_sim_scores = []
        for entry in similarity_scores:
            flatten_sim_scores.extend(entry.tolist())



        naive_sim_scores.append(flatten_sim_scores)
    naive_sim_scores=np.asarray(naive_sim_scores)
    if use_abs:
        naive_sim_scores=np.abs(naive_sim_scores)
    naive_sim_scores=np.mean(naive_sim_scores,axis=0)

    print('flatten_sim_scores',naive_sim_scores.shape,flush=True)
    return naive_sim_scores,valid_indices,valid_indices_2



# Define likelihood function for a single SNP pair
def likelihood(r_sq, p_A, p_B):
    """Computes negative log-likelihood for r² given allele frequencies."""

    # LD coefficient D
    D = np.sqrt(r_sq * p_A * (1 - p_A) * p_B * (1 - p_B))

    # Estimated haplotype frequencies
    p_AB = p_A * p_B + D
    p_Ab = p_A * (1 - p_B) - D
    p_aB = (1 - p_A) * p_B - D
    p_ab = (1 - p_A) * (1 - p_B) + D

    # Ensure probabilities are valid
    if (p_AB < 0 or p_AB > 1 or p_Ab < 0 or p_Ab > 1 or
            p_aB < 0 or p_aB > 1 or p_ab < 0 or p_ab > 1):
        return np.inf  # Return high cost for invalid solutions

    # Compute likelihood (avoid log(0))
    likelihood_value = p_AB * p_Ab * p_aB * p_ab
    if likelihood_value <= 0:
        return np.inf  # Prevent log(0)

    return -np.log(likelihood_value)  # Negative log-likelihood


def plot_correlation(gt, preds, mod_names, file_name):
    x_axis_name = 'gt'
    figsize=4
    fig, axes = plt.subplots(1, len(mod_names), figsize=(figsize*len(mod_names), figsize))
    print(np.mean(gt),np.median(gt),np.mean(np.abs(gt)),np.median(np.abs(gt)),np.min(gt),np.max(gt),np.min(np.abs(gt)),np.max(np.abs(gt)))
    for i,(pred,model_name) in enumerate(zip(preds,mod_names)):
        if len(mod_names)<=1:
            ax=axes
        else:
            ax=axes[i]
        print('DEBUG',model_name,pred[:5],len(pred),len(gt))
        dataFrame = pd.DataFrame({x_axis_name: gt, model_name: pred})
        print(dataFrame.columns,flush=True)
        sns.scatterplot(data=dataFrame, x=x_axis_name, y=model_name,ax=ax)
        sp = stats.spearmanr(gt, pred)[0]
        # Mean Absolute Error
        mae = np.mean(np.abs(gt - pred))
        rmse = np.sqrt(np.mean((gt - pred) ** 2))
        title=' Sp:'+str(np.round(sp,2))+' MAE:'+str(np.round(mae,3))+' RMSE:'+str(np.round(rmse,3))
        ax.set_title(title)

    # plt.show()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()
    plt.close()


def r2_random_from_marginals(pA, pB, rng=None):
    """
    Sample one random r^2 value within the feasible bounds
    given marginal allele frequencies.

    Parameters
    ----------
    pA, pB : 1D np.ndarray or array-like
        Allele frequencies for SNP1 and SNP2 (same shape).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        If None, np.random.default_rng() is used.

    Returns
    -------
    r2_samples : np.ndarray
        Random r^2 values, same shape as pA and pB.
        np.nan where undefined.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    pA = np.asarray(pA, dtype=float)
    pB = np.asarray(pB, dtype=float)

    if pA.shape != pB.shape:
        raise ValueError("pA and pB must have the same shape")

    denom = pA * (1 - pA) * pB * (1 - pB)

    Dmin = np.maximum(-pA * pB, -(1 - pA) * (1 - pB))
    Dmax = np.minimum(pA * (1 - pB), (1 - pA) * pB)

    r2_min = np.full(pA.shape, np.nan, dtype=float)
    r2_max = np.full(pA.shape, np.nan, dtype=float)

    valid = denom > 0
    r2_min[valid] = (Dmin[valid] ** 2) / denom[valid]
    r2_max[valid] = (Dmax[valid] ** 2) / denom[valid]

    low = np.minimum(r2_min[valid], r2_max[valid])
    high = np.maximum(r2_min[valid], r2_max[valid])



    r2_samples = np.full(pA.shape, np.nan, dtype=float)
    #r2_samples[valid] = rng.uniform(r2_min[valid], r2_max[valid])
    r2_samples[valid] = rng.uniform(low, high)
    mask = np.isnan(r2_samples)
    count_nans = np.sum(mask)

    print(count_nans)
    print(np.sum(valid))


    return r2_samples,valid


def read_LDx_results(file_path,position_keys):
    snp_dict = {}

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            # Skip empty lines
            if not line.strip():
                continue

            # Split by tab
            parts = line.strip().split("\t")

            try:
                # Required indices based on your description
                snp1 = parts[0]       # Location of SNP1
                snp2 = parts[1]       # Location of SNP2
                approx_mle = float(parts[12])  # Approx MLE estimate

                key = f"{snp1}_{snp2}"
                snp_dict[key] = approx_mle

            except (IndexError, ValueError) as e:
                print("Skipping line {line_num}: {e}")
    print('#### LDx entries:',len(snp_dict.keys()))
    ldx_mask=[]
    ldx_result=[]
    for pair1,pair2 in position_keys:
        if pair1 in snp_dict:
            ldx_result.append(snp_dict[pair1])
            ldx_mask.append(1)
        elif pair2 in snp_dict:
            ldx_result.append(snp_dict[pair2])
            ldx_mask.append(1)
        else:
            ldx_result.append(0)
            ldx_mask.append(0)
    ldx_mask=np.asarray(ldx_mask)>0.5
    ldx_result=np.asarray(ldx_result)
    print(ldx_result.shape,ldx_mask.shape,np.sum(ldx_mask))
    return ldx_result,ldx_mask


def calculate_corr_values(filtered_actual_linkage, filtered_pot_linkage):
    mae_model = np.mean(np.abs(filtered_actual_linkage - filtered_pot_linkage))
    rmse_model = np.sqrt(np.mean((filtered_actual_linkage - filtered_pot_linkage) ** 2))
    spear = stats.spearmanr(filtered_actual_linkage, filtered_pot_linkage)
    pears=stats.pearsonr(filtered_actual_linkage, filtered_pot_linkage)

    return spear[0],pears[0],mae_model, rmse_model



def read_LDx(file_paths,filter_name):
    snp_dict = {}
    for file_path in file_paths:
        file_path=file_path.replace('.txt','')+filter_name
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                # Skip empty lines
                if not line.strip():
                    continue

                # Split by tab
                parts = line.strip().split("\t")

                try:
                    # Required indices based on your description
                    snp1 = parts[0]       # Location of SNP1
                    snp2 = parts[1]       # Location of SNP2
                    approx_mle = float(parts[12])  # Approx MLE estimate

                    key = f"{snp1}_{snp2}"
                    if key not in snp_dict:
                        snp_dict[key]=[]
                    snp_dict[key].append(approx_mle)

                except (IndexError, ValueError) as e:
                    print("Skipping line {line_num}: {e}")
        print('#### LDx entries:',len(snp_dict.keys()))
    for key, value in snp_dict.items():
        snp_dict[key]=sum(value)/len(value)
    return snp_dict

def get_LDx_intersections(val_indices1,val_indices2, LDx_str_ps, distances):
    LDx_result_flat=[]
    LDx_result_mask=[]
    counter=[0,0]
    for i, idx in zip(val_indices1,val_indices2):
        #if idx < 50 or idx + 51 > distances.shape[0]:
        #    print('skip', idx, i)
        #    continue
        p0=distances[idx]
        for p1 in np.concatenate([distances[idx - 50:idx],distances[idx + 1:idx + 51]]):
            counter[0]+=1
            key0=str(p0)+'_'+str(p1)
            key1=str(p1)+'_'+str(p0)
            if key1 in LDx_str_ps:
                LDx_result_flat.append(LDx_str_ps[key1])
                LDx_result_mask.append(True)
                counter[1] += 1
            elif key0 in LDx_str_ps:
                LDx_result_flat.append(LDx_str_ps[key0])
                LDx_result_mask.append(True)
                counter[1] += 1
            else:
                LDx_result_flat.append(None)
                LDx_result_mask.append(False)
    print(counter)
    return np.asarray(LDx_result_flat),np.asarray(LDx_result_mask)





def linkage_correlation_filter(model_rows, afc_rows, linkage_rows,base_model_names, linkage_idx, num_neigbors, plot_path,num_submodels=3,
                               chrom='2L',debugg=0,use_abs=True,work_from='mogon',tmp_data_folder='',afc_gen=5,
                               num_train_gens=6,gt_ld_neighbors=50,use_LDx_separate=1,time_series_LD=False):
    sp_idx = 0
    matplotlib.rcParams['lines.markersize'] = 4

    for entry in base_model_names:
        print(entry,flush=True)
        (model_name, plot_name, time_bz, fine_tune, animal_noise, ds_name, animal, dataset_type, p_LD_influence)=entry
        result_dict = dict()

        eval_path = fix_variables[work_from][animal]['eval_positions']
        print(eval_path,flush=True)
        linkage_data = pkl.load(open(eval_path, 'rb'))
        if len(linkage_data['result'])==12:#isinstance(linkage_data,dict):
            idx_list_pkl_data = np.asarray(linkage_data['selected']).tolist() + np.asarray(linkage_data['unselected']).tolist()
            idx_list=list(range(len(idx_list_pkl_data)))
            [eval_positions, afc, linkage_def1, linkage_def_kramer, no_window_linkage_def, tmp_distance,
             afc_neighbor_metric, afc_neighbor_linkage_corr, all_n_f, r2, d_normal,r_no_sq] = linkage_data['result']
            linkage_trust_dict=np.asarray([])
        else:
            [eval_positions_, linkage_trust_dict, sf_filter, all_n_f, all_n_f_rsquare, all_n_f_D_normal,
             all_n_f_r_no_square]=linkage_data['result']
            idx_list_pkl_data = np.asarray(eval_positions_['selected'][chrom]).tolist() + np.asarray(eval_positions_['unselected'][chrom]).tolist()
            idx_list = list(range(len(idx_list_pkl_data)))
            r2=all_n_f_rsquare[chrom]
            d_normal= all_n_f_D_normal[chrom]
            all_n_f=all_n_f[chrom]
            r_no_sq=all_n_f_r_no_square[chrom]
            #linkage_trust_dict=np.asarray(linkage_trust_dict[chrom])
            linkage_trust_dict=linkage_trust_dict[chrom]
            #print('linkege trust dict shape',linkage_trust_dict.shape)

        #r2=np.asarray(r2)
        #print('Example r2',r2)
        #d_normal=np.asarray(d_normal)
        #all_n_f=np.asarray(all_n_f)
        #print('example d normal linkage',d_normal.shape,all_n_f.shape,d_normal[0])
        #d_normal_normalized=d_normal-np.min(d_normal)
        #d_normal_normalized=d_normal_normalized/np.max(d_normal_normalized)
        pkl_data=pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))
        gt_freqs = pkl_data['alleles']
        distances=pkl_data['distances']
        if isinstance(gt_freqs, dict):
            gt_freqs = np.squeeze(gt_freqs[chrom])
            distances=np.squeeze(distances[chrom])

        gt_freqs = np.transpose(gt_freqs, (2, 0, 1))  # [1:]

        # TODO naive linkage calculation with tf / numpy :)
        print('gt_freqs.shape,len(idx_list_pkl_data)', gt_freqs.shape, len(idx_list_pkl_data))


        naive_sim_scores, valid_indices, val_indices2 = calculate_naive_sim_scores(gt_freqs, idx_list_pkl_data, #gt_freqs
                                                                                   use_abs=use_abs,
                                                                                   num_train_gens=num_train_gens)
        #gt_freqs = np.arcsin(np.sqrt(gt_freqs)) / np.arcsin(np.sqrt(1))
        #idx_list_pkl_data=valid_indices
        #idx_list=val_indices2
        #linkage_trust_dict = linkage_trust_dict[val_indices2]
        linkage_trust_arr=[]


        if time_series_LD:
            #TODO
            ER_haplo_base_path = fix_variables[work_from][animal]['ER_haplos']
            out_file = ER_haplo_base_path + '/LD_ER_merge.pkl'
            dna_target_positions, dna_neigbor_positions, LD_result_merge = pkl.load(open(out_file, 'rb'))
            #(target_positions,4,100,generations,replicates)
            linkage_defs=np.mean(np.mean(LD_result_merge[:,linkage_idx,:,0:afc_gen+1],axis=2),axis=2)
        else:
            linkage_defs = [all_n_f, r2, d_normal,r_no_sq][linkage_idx]
        freqs_focus_once = gt_freqs[0, :, 0]  # shape (replicates,num_snps)
        afc = np.mean(np.abs(gt_freqs[0] - gt_freqs[afc_gen]), axis=1)



        freqs_focus = [] # shape (replicates,num_snpsx100)
        freqs_neighbors= [] # shape (replicates,num_snpsx100)
        afc_neighbor = []
        actual_linkage = []
        #TODO filter unvalid indices DONE
        val_indices2=[]
        val_indices1=[]
        for i, idx in enumerate(idx_list_pkl_data):
            if idx < 50 or idx + 51 > gt_freqs.shape[1]:
                print('skip',idx,i)
                continue
            afc_neighbor.extend(afc[idx - 50:idx].tolist() + afc[idx + 1:idx + 51].tolist())
            freqs_focus.append([freqs_focus_once[idx]]*100)
            freqs_neighbors.append(gt_freqs[0,idx - 50:idx,0])
            freqs_neighbors.append(gt_freqs[0,idx + 1:idx + 51,0])
            val_indices2.append(idx)
            val_indices1.append(i)

            entry=linkage_defs[i]
            actual_linkage.extend(entry.tolist())
            #columns_link = ['dataset_name', 'dataset_type', 'p_{LD} influence', 'Linkage']
            for e in entry:
                linkage_rows.append({
                    'dataset_name': ds_name,
                    'dataset_type': dataset_type,
                    'p_{LD} influence': p_LD_influence,
                    'Linkage': e,
                    'Chrom':chrom
                })
            if len(linkage_trust_dict):
                linkage_trust_arr.extend(linkage_trust_dict[i])
                if not len(linkage_trust_dict[i]):
                    print(i,idx,'ERROR nothing inside ld trust dict')
        idx_list = val_indices1
        afc_focus = np.abs(np.repeat(np.expand_dims(afc[val_indices2], axis=-1), 100, axis=-1)).flatten()
        linkage_trust_arr=np.asarray(linkage_trust_arr)
        actual_linkage = np.asarray(actual_linkage)
        print('actual_linkage',actual_linkage.shape,afc_focus.shape,linkage_trust_arr.shape)
        #############################################################################################

        afc_neighbor = np.abs(np.asarray(afc_neighbor))

        freqs_focus=np.concatenate(freqs_focus,axis=0)
        freqs_neighbors=np.concatenate(freqs_neighbors,axis=0)

        # these lines are just for debugging:
        #freqs_focus =freqs_focus[:100]
        #freqs_neighbors=freqs_neighbors[:100]

        #'''
        if linkage_idx==1:
            num_cores = cpu_count()
            print(f"Using {num_cores} CPU cores for parallel execution.")
            # Function to optimize r² for a single SNP pair
            def optimize_r_squared(i):
                """Optimizes r² for a given SNP index."""
                result = minimize(likelihood, x0=[0.1], args=(p_A[i], p_B[i]), bounds=[(0, 1)])
                return result.x[0]  # Extract optimized r²

            num_snps=freqs_focus.shape[0]
            print('start LDX')
            num_cores = -1  # Uses all available cores
            p_A, p_B = freqs_focus, freqs_neighbors
            random_r_square_analytical,valid_positions = r2_random_from_marginals(p_A, p_B)


            #if use_LDx_separate:
            if 'LDx' in fix_variables[work_from][animal_noise]:
            #if False:#'LDx_gt' in fix_variables[work_from][animal]:

                '''
                LDx_path_gt = fix_variables[work_from][animal]['LDx_gt']
                (pos_pair_list, LDx_estimate, LD_gt)=pkl.load(open(LDx_path_gt,'rb'))
                LDx_gt_link_x_s, LDx_gt_r_square, LDx_gt_normal_D, LDx_gt_r_norm, LDx_gt_trust=zip(*LDx_estimate)
                LDx_gt_r_square=np.asarray(LDx_gt_r_square)
                
                # TODO build a mask threshold:

                LDx_gt_afc_focus,LDx_gt_afc_neighbor=calculate_afc(pos_pair_list,distances,afc)
                for afc_diff_idx in range(20):

                    afc_diff = afc_diff_idx * 0.01
                    #if len(linkage_trust_dict):

                    mask = ((LDx_gt_afc_focus >= afc_diff) & (LDx_gt_afc_neighbor >= afc_diff)) & (linkage_trust_arr == 1.0)
                    #else:
                    #    mask = ((afc_focus >= afc_diff) & (afc_neighbor >= afc_diff))  # & (actual_linkage < 0.999)
                    num_true = np.count_nonzero(mask)

                    if num_true < 30:
                        break
                '''
                #LDx_paths = fix_variables[work_from][animal]['LDx'][chrom]
                #LDx_str_ps = read_LDx(LDx_paths)
                #LDx_gt_r_square, LDx_result_mask = get_LDx_intersections(val_indices1,val_indices2, LDx_str_ps, distances)

                for afc_diff_idx in range(20):
                #for afc_diff , filter_name in enumerate(['.txt','_filter.txt']):
                    LDx_paths = fix_variables[work_from][animal_noise]['LDx'][chrom]
                    #LDx_str_ps = read_LDx(LDx_paths,filter_name)#'.txt'
                    LDx_str_ps = read_LDx(LDx_paths,'.txt')  #
                    LDx_gt_r_square, LDx_result_mask = get_LDx_intersections(val_indices1, val_indices2, LDx_str_ps,
                                                                             distances)
                    afc_diff = afc_diff_idx * 0.01
                    #afc_diff = afc_diff_idx * 0.01
                    if len(linkage_trust_dict):
                        #mask =  (linkage_trust_arr == 1.0) & LDx_result_mask
                        mask = ((afc_focus >= afc_diff) & (afc_neighbor >= afc_diff)) & (linkage_trust_arr == 1.0) & LDx_result_mask
                    else:
                        #mask = LDx_result_mask  # & (actual_linkage < 0.999)
                        mask = ((afc_focus >= afc_diff) & (afc_neighbor >= afc_diff))& LDx_result_mask  # & (actual_linkage < 0.999)
                    num_true = np.count_nonzero(mask)

                    if num_true < 30:
                        break
                    print(actual_linkage.shape,LDx_gt_r_square.shape,mask.shape)
                    sp, pears_ldx, mae_ldx, rmse_ldx = calculate_corr_values(actual_linkage[mask],LDx_gt_r_square[mask])#np.asarray(r2).flatten()[mask]

                    afc_rows.append({
                        'dataset_name': ds_name,
                        'dataset_type': dataset_type,
                        'p_{LD} influence': p_LD_influence,
                        'AFC filter': afc_diff,
                        'Spearman roh': sp,
                        'Pearson': pears_ldx,
                        'LD estimator': 'LDx',
                        'Chrom': chrom,
                        'MAE': mae_ldx,
                        'RMSE': rmse_ldx,
                        'num_eval_spns':num_true,
                        'num_all_spns':np.count_nonzero( (linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else actual_linkage.shape[0]

                    })
                    ####################################################################################
                    # make a plot for debugging:
                    if True:#(not afc_diff_idx):
                        preds = [LDx_gt_r_square[mask]]
                        mod_names = ['LDx']

                        plot_correlation(gt=actual_linkage[mask],
                                         preds=preds,
                                         mod_names=mod_names,
                                         file_name=plot_path + '_' + str(linkage_idx) + '_' + str(chrom) + '_' + str(
                                             ds_name) + '_' + str(p_LD_influence) + '-LDX-' +'filter_idx' + str(afc_diff) + 'LD_corr_debug.png')

                    #######################################################################################
                    print('LDx num true', afc_diff, num_true,sp)



            if False:#'LDx' in fix_variables[work_from][animal]:
                position_keys = []
                for idx1 in idx_list_pkl_data:
                    # Loop over neighbors within ±50 positions, excluding idx itself
                    for idx2 in list(range(idx1 - 50, idx1)) + list(range(idx1 + 1, idx1 + 51)):
                        position_keys.append(
                            (f"{distances[idx1]}_{distances[idx2]}", f"{distances[idx2]}_{distances[idx1]}"))
                LDx,Ldx_mask=read_LDx_results(fix_variables[work_from][animal]['LDx'],position_keys)
            else:
                LDx, Ldx_mask = np.asarray([]),np.asarray([])
            #r_squared_mle = Parallel(n_jobs=num_cores)(delayed(optimize_r_squared)(i) for i in range(num_snps))


            # Convert results to NumPy array
            #LDx_sim_scores = np.array(r_squared_mle)
            #print('LDx_sim_scores.shape',LDx_sim_scores.shape)
        #'''


        print(afc_neighbor.shape,afc_focus.shape)
        #direction_similarity = np.abs(afc_neighbor - afc_focus)

        mask_arr = []
        for afc_diff_idx in range(20):

            afc_diff = afc_diff_idx * 0.01
            if afc_diff_idx not in result_dict:
                result_dict[afc_diff_idx] = dict()
            if len(linkage_trust_dict):
                # TODO filter unvalid indices DONE
                print(afc_neighbor.shape, afc_focus.shape,linkage_trust_arr.shape,len(val_indices2),len(idx_list_pkl_data),len(linkage_trust_dict))
                mask = ((afc_focus >= afc_diff) & (afc_neighbor >= afc_diff)) & (linkage_trust_arr==1.0)
            else:
                mask = ((afc_focus >= afc_diff) & (afc_neighbor >= afc_diff)) # & (actual_linkage < 0.999)
            num_true = np.count_nonzero(mask)

            if num_true < 30:
                break
            mask_arr.append(mask)
            print('mask size',num_true, end='\t')
        #######################################################################################################



        for afc_diff_idx in range(len(mask_arr)):
            result_dict[afc_diff_idx][plot_name + ' ' + ds_name] = np.zeros((num_submodels, 4))



        for submodel_num in range(num_submodels):
            model_name_ = model_name + '_' + str(submodel_num) + '_fine'
            path = tmp_data_folder + 'models/' + model_name_ + '/eval_data/train_data.pkl'
            in_stream = open(path, 'rb')
            data = pkl.load(in_stream)

            num_neighbors=50#json_data['num_neighbors']

            potential_linkage_data = data['window_representation']
            if isinstance(potential_linkage_data, dict):
                potential_linkage_data=potential_linkage_data[chrom] #shape p.zeros((1,len(idx_list), sample_size, 1+(dna_vae.snp_window_half*2)))
            print('model values', potential_linkage_data.shape,len(idx_list),idx_list[:5],chrom,flush=True )
            in_stream.close()
            if use_abs:
                #pot_linkage=np.abs(pot_linkage)
                print('take model abs values',potential_linkage_data.shape,)
                potential_linkage_data = np.mean(np.abs(np.squeeze(potential_linkage_data)[idx_list]), axis=1)
            else:
                potential_linkage_data=np.mean(np.squeeze(potential_linkage_data)[idx_list],axis=1)
            # shape (#snps,neighbors)
            neighbor_diff=gt_ld_neighbors-num_neighbors
            if potential_linkage_data.shape[1]%2==1:
                pot_linkage = []
                if neighbor_diff == 0:
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry[:50].tolist())
                        pot_linkage.extend(entry[51:].tolist())
                elif neighbor_diff < 0:
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry[-neighbor_diff:-neighbor_diff+gt_ld_neighbors].tolist())
                        pot_linkage.extend(entry[-neighbor_diff+gt_ld_neighbors+1:-neighbor_diff+(2*gt_ld_neighbors)+1].tolist())
                else:
                    # num_neighbors is smaler
                    print('not implemented - as case does not exist so far')
                    pass

                pot_linkage=np.asarray(pot_linkage)
            else:
                pot_linkage = []
                if neighbor_diff==0:
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry.tolist())
                elif neighbor_diff<0:
                    #num_neighbors is larger
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry.tolist()[-neighbor_diff:-neighbor_diff+(2*gt_ld_neighbors)])
                else:
                    # num_neighbors is smaler
                    print('not implemented - as case does not exist so far')
                    pass


                pot_linkage = np.asarray(pot_linkage)



            for afc_diff_idx in range(len(mask_arr)):

                correlations=[]
                filtered_pot_linkage=pot_linkage[mask_arr[afc_diff_idx]]
                filtered_actual_linkage = actual_linkage[mask_arr[afc_diff_idx]]
                #filtered_direction_similarity = direction_similarity[mask_arr[afc_diff_idx]]
                # small debugging plot
                res_1, pears, mae_model, rmse_model = calculate_corr_values(filtered_actual_linkage,filtered_pot_linkage)
                corr_baseline, pears_base, mae_base, rmse_base = calculate_corr_values(filtered_actual_linkage, naive_sim_scores[mask_arr[afc_diff_idx]])

                correlations.extend([0,res_1,0,corr_baseline])

                result_dict[afc_diff_idx][plot_name + ' ' + ds_name][submodel_num]=np.asarray(correlations)

                #####################################################################################
                # debugging histplot
                if (not afc_diff_idx) or afc_diff_idx==(len(mask_arr)-3):
                    preds=[naive_sim_scores[mask_arr[afc_diff_idx]],filtered_pot_linkage]
                    mod_names=['Baseline','VAE']

                    plot_correlation(gt=filtered_actual_linkage,
                                     preds=preds,
                                     mod_names=mod_names,
                                     file_name=plot_path+'_'+str(linkage_idx)+'_'+str(chrom)+'_'+str(ds_name)+'_'+str(p_LD_influence)+'-submod'+str(submodel_num)+'filter_idx'+str(afc_diff_idx)+'LD_corr_debug.png')

                # save values
                afc_diff = afc_diff_idx * 0.01

                model_rows.append({
                    'dataset_name':ds_name,
                    'dataset_type':dataset_type,
                    'p_{LD} influence':p_LD_influence,
                    'model name':model_name,
                    'model plot name':plot_name,
                    'submodel_num':submodel_num, 'AFC filter':afc_diff,
                    'Spearman roh':res_1,
                    'Pearson':pears,
                    'LD estimator':'VAE',
                    'Chrom':chrom,
                    'MAE':mae_model,
                    'RMSE':rmse_model,
                    'num_eval_spns':  np.count_nonzero(mask_arr[afc_diff_idx]),
                    'num_all_spns': np.count_nonzero((linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else actual_linkage.shape[0]

                })


                afc_rows.append({
                    'dataset_name': ds_name,
                    'dataset_type': dataset_type,
                    'p_{LD} influence': p_LD_influence,
                    'AFC filter': afc_diff,
                    'Spearman roh': corr_baseline,
                    'Pearson': pears_base,
                    'LD estimator': 'org_sim',
                    'Chrom':chrom,
                    'MAE': mae_base,
                    'RMSE': rmse_base,
                    'num_eval_spns':  np.count_nonzero(mask_arr[afc_diff_idx]),
                    'num_all_spns': np.count_nonzero((linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else actual_linkage.shape[0]

                })

                if linkage_idx==1:
                    #LDx,Ldx_mask
                    if LDx.shape[0]:
                        combi_filter = mask_arr[afc_diff_idx] & Ldx_mask
                        if np.sum(combi_filter)>3:

                            sp, pears_ldx, mae_ldx, rmse_ldx = calculate_corr_values(actual_linkage[combi_filter] ,LDx[combi_filter])

                            afc_rows.append({
                                'dataset_name': ds_name,
                                'dataset_type': dataset_type,
                                'p_{LD} influence': p_LD_influence,
                                'AFC filter': afc_diff,
                                'Spearman roh':sp,
                                'Pearson': pears_ldx,
                                'LD estimator': 'LDx subset',
                                'Chrom':chrom,
                                'MAE':mae_ldx,
                                'RMSE':rmse_ldx,
                    'num_eval_spns':  np.count_nonzero(combi_filter),
                    'num_all_spns': np.count_nonzero((linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else actual_linkage.shape[0]

                            })

                    #random_r_square_analytical
                    combi_filter=mask_arr[afc_diff_idx]&valid_positions
                    if np.sum(combi_filter)>3:
                        sp, pears_ldx, mae_ldx, rmse_ldx = calculate_corr_values(actual_linkage[mask_arr[afc_diff_idx]&valid_positions],random_r_square_analytical[mask_arr[afc_diff_idx]&valid_positions])
                        afc_rows.append({
                            'dataset_name': ds_name,
                            'dataset_type': dataset_type,
                            'p_{LD} influence': p_LD_influence,
                            'AFC filter': afc_diff,
                            'Spearman roh':sp,
                            'Pearson': pears_ldx,
                            'LD estimator': 'random',
                            'Chrom':chrom,
                            'MAE':mae_ldx,
                            'RMSE':rmse_ldx,
                    'num_eval_spns':  np.count_nonzero(combi_filter),
                    'num_all_spns': np.count_nonzero((linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else actual_linkage.shape[0]

                        })


        print('plot data')
        #if debugg:
        x_axis_name = "Percentage considered SNPs with corresponding AFC filter"
        y_axis_name= 'Spearman Rho'
        hue_name='LD estimator'
        style_name='Relation type'

        x_axis_data=[]
        y_axis_data=[]
        hue_data=[]
        style_data=[]

        for afc_diff_idx,patrial_result_dict in result_dict.items():
            afc_diff=afc_diff_idx*0.01
            for m_idx,(model_name_ds, corr_data ) in enumerate(patrial_result_dict.items()):
                #if not m_idx:
                    # save afc result:

                num_true = np.count_nonzero(mask_arr[afc_diff_idx])
                all = np.count_nonzero(mask_arr[0])
                percentage= num_true / all
                x_ax_label=str(round(percentage,4))+'\n'+str(round(afc_diff,2))

                x_axis_data.append(x_ax_label)
                y_axis_data.append(corr_data[0,3])
                hue_data.append('Baseline')
                style_data.append('linear')
                for sub_m in range(num_submodels):

                    x_axis_data.append(x_ax_label)
                    y_axis_data.append(corr_data[sub_m,1])
                    hue_data.append('VAE')
                    style_data.append('linear')



        fig, axis = plt.subplots(1, 1, figsize=(7, 3.5))

        dataFrame = pd.DataFrame(
            {x_axis_name:x_axis_data, y_axis_name: y_axis_data,hue_name:hue_data})
        default_palette = sns.color_palette("tab10")

        custom_palette = {'VAE': default_palette[0],  # Orange
                          'LDx': default_palette[2],  # Green
                          'Baseline': default_palette[1]
                          }  # Blue
        sns.lineplot(data=dataFrame, x=x_axis_name, y=y_axis_name,hue=hue_name, ax=axis,palette=custom_palette)
        #axis[1, 1].set_title('filter(AFC diff vs D) ' + str(np.round(res_3.statistic, 4)))

        plt.tight_layout()
        plt.savefig(plot_path+model_name+chrom+'LD_spearman_corr_perc.png')
        #plt.show()
    return model_rows,afc_rows,linkage_rows








