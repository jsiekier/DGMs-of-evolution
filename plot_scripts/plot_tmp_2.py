
import os
import pickle as pkl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from variables import fix_variables
from scipy import stats
import time
import tensorflow as tf
from eval.evaluation_metrics import all_metrics, all_metrics_list
import matplotlib
from scipy.optimize import minimize
from joblib import Parallel, delayed,cpu_count

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



def load_model_data_noise_less_info(entry,summary_method=1,chrom='2L'):
    #model_name, plot_name, time_bz, _, animal_noise, ds_name, animal=entry
    (model_name, plot_name, time_bz, fine_tune, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
    eval_path = fix_variables[work_from][animal]['eval_positions']
    print(eval_path)
    linkage_data = pkl.load(open(eval_path, 'rb'))
    num_targets=int(len(linkage_data['selected'])/100)
    print('--------num targets VAE',num_targets)
    real_selected_points = [linkage_data['selected'][50 + (s_idx * 100)] for s_idx in range(num_targets)]

    idx_lists = [linkage_data['unselected'], real_selected_points]#[[0,1,2,4],[0,1,2,3,4]]#
    idx_list_model_data=[list(range(len(linkage_data['selected']),len(linkage_data['selected'])+len(linkage_data['unselected']))),
                         [list(range(len(linkage_data['selected'])))[50 + (s_idx * 100)] for s_idx in range(num_targets)]]


    noise_freqs = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        noise_freqs = noise_freqs[chrom]
    noise_freqs = np.transpose(noise_freqs, (2, 0, 1))[1:]
    gt_freqs = pkl.load(open(fix_variables[work_from][animal]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        gt_freqs = gt_freqs[chrom]
    gt_freq = np.transpose(gt_freqs, (2, 0, 1))[1:]

    mean_gt = np.mean(gt_freq, axis=2)
    std_gt = np.std(gt_freq, axis=2)

    #####################################################################

    # remember we start idx with gen 1
    noise_freqs = noise_freqs[5]



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


        remaining_results = {'std': [np.mean(np.abs(std_gt[time_bz - 1:,idx_lists[0]] - std_pred[:,idx_list_model_data[0]]), axis=1),
                                     np.mean(np.abs(std_gt[time_bz - 1:,idx_lists[1]] - std_pred[:,idx_list_model_data[1]]), axis=1)],

                             'mean':  [np.mean(np.abs(mean_gt[time_bz - 1:,idx_lists[0]] - mean_pred[:,idx_list_model_data[0]]), axis=1),
                                     np.mean(np.abs(mean_gt[time_bz - 1:,idx_lists[1]] - mean_pred[:,idx_list_model_data[1]]), axis=1)],}
        for metric in all_metrics_list:
            remaining_results[metric.internal_name] = []


        prediction = np.transpose(prediction, (0, 2, 1))





        for idx_list_gt,idx_list_model in zip(idx_lists,idx_list_model_data):
            for metric in all_metrics_list:
                remaining_results[metric.internal_name].append(np.zeros((prediction.shape[0])))
            for gen in range(prediction.shape[0]):
                for metric in all_metrics[3]:

                    remaining_results[metric.internal_name][-1][gen] = metric.eval_function(
                        gt_freq[gen + (time_bz - 1), idx_list_gt], prediction[gen, idx_list_model], noise_freqs[idx_list_gt],
                        summary_method)
        in_stream.close()
        plot_data.append(remaining_results)
    return plot_data
def read_wf_file_noise_less_info(animal_noise,animal,summary_method=1,chrom='2L'):
    num_train_it=0

    wf_file_name = tmp_data_folder + 'wf_data/wf_simulations_' + animal_noise + '_' + str(num_train_it) + wf_file_end
    print('WF file name',wf_file_name)
    in_stream = open(wf_file_name, 'rb')
    wf_data = pkl.load(in_stream)
    in_stream.close()
    gt_data = dict()
    ####################################################################################################
    eval_path = fix_variables[work_from][animal]['eval_positions']
    print(eval_path)
    linkage_data = pkl.load(open(eval_path, 'rb'))
    num_targets=int(len(linkage_data['selected'])/100)
    print('--------num targets',num_targets)
    real_selected_points = [linkage_data['selected'][50 + (s_idx * 100)] for s_idx in range(num_targets)]
    idx_lists = [linkage_data['unselected'], real_selected_points]#[[0,1,2,4],[0,1,2,3,4]]#
    idx_list_model_data=[list(range(len(linkage_data['selected']),len(linkage_data['selected'])+len(linkage_data['unselected']))),
                         [list(range(len(linkage_data['selected'])))[50 + (s_idx * 100)] for s_idx in range(num_targets)]]
    print('selected positions',real_selected_points)


    ####################################################################################################
    noise_freqs = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        noise_freqs = noise_freqs[chrom]
    noise_freqs = np.transpose(noise_freqs, (2, 0, 1))[1:]
    gt_stuff=pkl.load(open(fix_variables[work_from][animal]['sync_process'], 'rb'))
    gt_freqs = gt_stuff['alleles']
    position_data=gt_stuff['distances']

    if chrom != '':
        gt_freqs = gt_freqs[chrom]
        position_data=np.squeeze(position_data[chrom])
    print('selected positions', position_data[real_selected_points])
    gt_freq = np.transpose(gt_freqs, (2, 0, 1))[1:]

    mean_gt = np.mean(gt_freq, axis=2)
    std_gt = np.std(gt_freq, axis=2)

    #####################################################################

    # remember we start idx with gen 1
    noise_freqs = noise_freqs[5]

    prediction = wf_data['freqs'][chrom]


    std_pred = np.std(prediction, axis=(0, 2))[1:]

    mean_pred = np.mean(prediction, axis=(0, 2))[1:]

    prediction = np.transpose(prediction, (1, 3, 0, 2))
    prediction = np.reshape(prediction,(prediction.shape[0], prediction.shape[1], prediction.shape[2] * prediction.shape[3]))

    prediction = prediction[1:]
    print('DEBUG shapes of prediction:', prediction.shape, 'gt freq shape', gt_freq.shape)

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
                 np.mean(np.abs(mean_gt[:, idx_lists[1]] - mean_pred[:, idx_list_model_data[1]]), axis=1)], }

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

    return remaining_results
def make_sum_paper_plot_no_wf(tmp_data_folder, input_data, num_submodels, finetuning, plot_path='',chrom='2L'):
    max_time_bz=6
    num_generations=16
    cut_time = 1
    columns = ['dataset_name', 'snp type', 'submodel_num','result_metric','result_metric_name','dataset_type','p_{LD} influence','model name']
    df = pd.DataFrame(columns=columns)


    all_noise_ds=dict()
    for entry in input_data:
        (model_name, plot_name, time_bz, fine_tune, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
        all_noise_ds[animal_noise+'\t'+animal]=[ds_name,dataset_type,p_LD_influence]
    for key,values in all_noise_ds.items():
        animal_noise,animal=key.split('\t')
        ds_name, dataset_type, p_LD_influence=values
        # now calculate wf stuff :)
        print([ds_name,dataset_type,p_LD_influence])
        plot_data=read_wf_file_noise_less_info(animal_noise,animal,summary_method=1,chrom=chrom)
        for metric_name in ['relative_mean_error', 'relative_std_error','mean','std']:
            for snp_idx, snp_type in enumerate([ 'unselected','selected']):
                if metric_name=='relative_mean_error' and snp_idx==1:
                    print('WF',np.mean(plot_data[metric_name][snp_idx][max_time_bz+cut_time-1:num_generations], axis=0),
                          plot_data[metric_name][snp_idx][max_time_bz+cut_time-1: num_generations])
                result_data_gens = np.mean(plot_data[metric_name][snp_idx][max_time_bz+cut_time-1:num_generations], axis=0)
                meta_data = (ds_name, snp_type, 0, result_data_gens, metric_name, dataset_type, p_LD_influence, 'WF model')
                row = {columns[i]: meta_data[i] for i in range(len(columns))}
                # Append the dictionary as a new row to the DataFrame
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)




    for ax_i, entry in enumerate(input_data):

        print(ax_i, 'load model',entry, flush=True)
        plot_data = load_model_data_noise_less_info(entry,  summary_method=1,chrom=chrom)
        #plot_data=np.asarray(plot_data)
        #plot_data_ds.append(plot_data)
        tbs=entry[2]
        train_gens=7
        cut_time=train_gens-tbs
        print(entry[-2],end='\t')
        for submodel_num in range(num_submodels):
            submodel_data=plot_data[submodel_num]
            for metric_name in ['relative_mean_error','relative_std_error','mean','std']:
                for snp_idx, snp_type in enumerate(['unselected','selected']):
                    if metric_name == 'relative_mean_error' and snp_idx == 1:
                        print('VAE',np.mean(submodel_data[metric_name][snp_idx][cut_time:num_generations],axis=0),
                              metric_name,submodel_data[metric_name][snp_idx][cut_time:num_generations]
                              )
                    result_data_gens=np.mean(submodel_data[metric_name][snp_idx][cut_time:num_generations],axis=0)
                    meta_data=(entry[-4],snp_type,submodel_num,result_data_gens,metric_name,entry[-2],entry[-1],entry[1])
                    row = {columns[i]: meta_data[i] for i in range(len(columns))}
                    # Append the dictionary as a new row to the DataFrame
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)


    # save the df:
    pkl.dump(df,open(plot_path+'simulation_results.pkl','wb'))
    print('saved the simulation performance')


def calculate_naive_sim_scores(gt_freqs, idx_list_pkl_data):
    gt_freqs=np.transpose(gt_freqs,(1,0,2))#(2,0,1))
    print(len(idx_list_pkl_data))
    print(gt_freqs.shape,flush=True)
    naive_sim_scores=[]

    focal_snp=gt_freqs[idx_list_pkl_data,0:6]
    neighbors=[]
    for idx in idx_list_pkl_data:
        #[snp_idx - num_neighbors: snp_idx + num_neighbors + 1]
        neighbors.append(np.concatenate([gt_freqs[idx-50:idx,0:6],gt_freqs[idx+1:idx+51,0:6]],axis=0))
    neighbors=np.asarray(neighbors)


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
    naive_sim_scores=np.abs(naive_sim_scores)
    naive_sim_scores=np.mean(naive_sim_scores,axis=0)

    print('flatten_sim_scores',naive_sim_scores.shape,flush=True)
    return naive_sim_scores




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

def linkage_correlation_filter(base_model_names, linkage_idx, num_neigbors, plot_path,num_submodels=3,chrom='2L',debugg=0):
    sp_idx = 0
    matplotlib.rcParams['lines.markersize'] = 4
    #num_focus_snps=-1

    linkage_rows = []
    afc_rows=[]
    model_rows=[]
    for entry in base_model_names:
        print(entry,flush=True)
        (model_name, plot_name, time_bz, fine_tune, animal_noise, ds_name, animal, dataset_type, p_LD_influence)=entry
        result_dict = dict()

        eval_path = fix_variables[work_from][animal]['eval_positions']
        print(eval_path,flush=True)
        linkage_data = pkl.load(open(eval_path, 'rb'))
        idx_list_pkl_data = np.asarray(linkage_data['selected']).tolist() + np.asarray(linkage_data['unselected']).tolist()
        idx_list=list(range(len(idx_list_pkl_data)))


        [eval_positions, afc, linkage_def1, linkage_def_kramer, no_window_linkage_def, tmp_distance,
         afc_neighbor_metric, afc_neighbor_linkage_corr, all_n_f, r2, d_normal] = linkage_data['result']
        r2=np.asarray(r2)
        #print('Example r2',r2)





        d_normal=np.asarray(d_normal)
        d_normal_normalized=d_normal-np.min(d_normal)
        d_normal_normalized=d_normal_normalized/np.max(d_normal_normalized)
        linkage_defs = [all_n_f, r2, d_normal][linkage_idx]

        actual_linkage = []
        print('linkage_defs',linkage_defs.shape)
        for entry in linkage_defs:
            actual_linkage.extend(entry.tolist())
            #columns_link = ['dataset_name', 'dataset_type', 'p_{LD} influence', 'Linkage']
            for e in entry:
                linkage_rows.append({
                    'dataset_name': ds_name,
                    'dataset_type': dataset_type,
                    'p_{LD} influence': p_LD_influence,
                    'Linkage': e
                })

        actual_linkage = np.asarray(actual_linkage)
        print('actual_linkage',actual_linkage.shape)
        #############################################################################################
        gt_freqs = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))['alleles']
        if isinstance(gt_freqs, dict):
            gt_freqs = gt_freqs[chrom]

        gt_freqs = np.transpose(gt_freqs, (2, 0, 1))  # [1:]
        # TODO naive linkage calculation with tf / numpy :)
        naive_sim_scores=calculate_naive_sim_scores(gt_freqs,idx_list_pkl_data)
        print(gt_freqs.shape)
        freqs_focus_once=gt_freqs[0,:,0] # shape (replicates,num_snps)


        afc = np.mean(np.abs(gt_freqs[0] - gt_freqs[5]), axis=1)

        afc_focus = np.abs(np.repeat(np.expand_dims(afc[idx_list_pkl_data], axis=-1), 100, axis=-1)).flatten()
        freqs_focus = [] # shape (replicates,num_snpsx100)
        freqs_neighbors= [] # shape (replicates,num_snpsx100)
        afc_neighbor = []
        for idx in idx_list_pkl_data:
            afc_neighbor.extend(afc[idx - 50:idx].tolist() + afc[idx + 1:idx + 51].tolist())
            freqs_focus.append([freqs_focus_once[idx]]*100)
            freqs_neighbors.append(gt_freqs[0,idx - 50:idx,0])
            freqs_neighbors.append(gt_freqs[0,idx + 1:idx + 51,0])
        afc_neighbor = np.abs(np.asarray(afc_neighbor))

        freqs_focus=np.concatenate(freqs_focus,axis=0)
        freqs_neighbors=np.concatenate(freqs_neighbors,axis=0)


        if linkage_idx==1:
            num_cores = cpu_count()
            print(f"Using {num_cores} CPU cores for parallel execution.")
            # Function to optimize r² for a single SNP pair
            def optimize_r_squared(i):
                """Optimizes r² for a given SNP index."""
                result = minimize(likelihood, x0=[0.1], args=(p_A[i], p_B[i]), bounds=[(0, 1)])
                return result.x[0]  # Extract optimized r²

            num_snps=freqs_focus.shape[0]
            #batchsize=num_snps//100
            #print('batchsize',batchsize)

            #LDx_sim_scores=[]
            print('start LDX')
            #time_start = time.time()
            # Run optimization in parallel using all available CPU cores
            num_cores = -1  # Uses all available cores
            p_A, p_B = freqs_focus, freqs_neighbors
            r_squared_mle = Parallel(n_jobs=num_cores)(delayed(optimize_r_squared)(i) for i in range(num_snps))

            # Convert results to NumPy array
            LDx_sim_scores = np.array(r_squared_mle)
            print('LDx_sim_scores.shape',LDx_sim_scores.shape)



        print(afc_neighbor.shape,afc_focus.shape)
        direction_similarity = np.abs(afc_neighbor - afc_focus)

        mask_arr = []
        for afc_diff_idx in range(20):

            afc_diff = afc_diff_idx * 0.01
            if afc_diff_idx not in result_dict:
                result_dict[afc_diff_idx] = dict()
            mask = ((afc_focus >= afc_diff) & (afc_neighbor >= afc_diff))  # & (actual_linkage < 0.999)
            num_true = np.count_nonzero(mask)

            if num_true < 30:
                break
            mask_arr.append(mask)
            print(num_true, end='\t')
        #######################################################################################################



        for afc_diff_idx in range(len(mask_arr)):
            result_dict[afc_diff_idx][plot_name + ' ' + ds_name] = np.zeros((num_submodels, 4))


        for submodel_num in range(num_submodels):
            model_name_ = model_name + '_' + str(submodel_num) + '_fine'
            path = tmp_data_folder + 'models/' + model_name_ + '/eval_data/train_data.pkl'
            in_stream = open(path, 'rb')
            data = pkl.load(in_stream)
            potential_linkage_data = data['window_representation']
            if isinstance(potential_linkage_data, dict):
                potential_linkage_data=potential_linkage_data[chrom] #shape p.zeros((1,len(idx_list), sample_size, 1+(dna_vae.snp_window_half*2)))
            in_stream.close()
            potential_linkage_data=np.mean(np.squeeze(potential_linkage_data)[idx_list],axis=1)
            # shape (#snps,neighbors)
            pot_linkage = []
            for entry in potential_linkage_data:
                pot_linkage.extend(entry[:50].tolist())
                pot_linkage.extend(entry[51:].tolist())

            pot_linkage=np.asarray(pot_linkage)
            #if not dataset_type=='(100,40)':
            pot_linkage=np.abs(pot_linkage)
            print('take model abs values')

            for afc_diff_idx in range(len(mask_arr)):

                correlations=[]
                filtered_pot_linkage=pot_linkage[mask_arr[afc_diff_idx]]
                filtered_actual_linkage = actual_linkage[mask_arr[afc_diff_idx]]
                filtered_direction_similarity = direction_similarity[mask_arr[afc_diff_idx]]
                # small debugging plot

                res_0=[0]#poly_correlation(filtered_actual_linkage, filtered_pot_linkage)
                res_1 = stats.spearmanr(filtered_actual_linkage, filtered_pot_linkage)
                res_2 =[0] #poly_correlation(filtered_actual_linkage, filtered_direction_similarity)
                res_3 = stats.spearmanr(filtered_actual_linkage, filtered_direction_similarity)

                correlations.extend([res_0[sp_idx],res_1[sp_idx],res_2[sp_idx],res_3[sp_idx]])

                result_dict[afc_diff_idx][plot_name + ' ' + ds_name][submodel_num]=np.asarray(correlations)


                # save values
                afc_diff = afc_diff_idx * 0.01

                model_rows.append({
                    'dataset_name':ds_name,
                    'dataset_type':dataset_type,
                    'p_{LD} influence':p_LD_influence,
                    'model name':model_name,
                    'model plot name':plot_name,
                    'submodel_num':submodel_num, 'AFC filter':afc_diff,
                    'Spearman roh':res_1[0],
                    'LD estimator':'VAE'

                })

                afc_rows.append({
                    'dataset_name': ds_name,
                    'dataset_type': dataset_type,
                    'p_{LD} influence': p_LD_influence,
                    'AFC filter': afc_diff,
                    'Spearman roh': res_3[0],
                    'LD estimator':'afc'

                })
                afc_rows.append({
                    'dataset_name': ds_name,
                    'dataset_type': dataset_type,
                    'p_{LD} influence': p_LD_influence,
                    'AFC filter': afc_diff,
                    'Spearman roh': stats.spearmanr(filtered_actual_linkage, naive_sim_scores[mask_arr[afc_diff_idx]])[0],
                    'LD estimator': 'org_sim'

                })
                if linkage_idx==1:

                    sp=stats.spearmanr(filtered_actual_linkage, LDx_sim_scores[mask_arr[afc_diff_idx]])
                    print('LDx Spearman',sp[0],afc_diff_idx)

                    afc_rows.append({
                        'dataset_name': ds_name,
                        'dataset_type': dataset_type,
                        'p_{LD} influence': p_LD_influence,
                        'AFC filter': afc_diff,
                        'Spearman roh':sp[0],
                        'LD estimator': 'LDx'

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
                    y_axis_data.append(corr_data[0,1])
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
        sns.lineplot(dataFrame, x=x_axis_name, y=y_axis_name,hue=hue_name, ax=axis,palette=custom_palette)
        #axis[1, 1].set_title('filter(AFC diff vs D) ' + str(np.round(res_3.statistic, 4)))

        plt.tight_layout()
        plt.savefig(plot_path+model_name+'LD_spearman_corr_perc.png')
        #plt.show()
    df_model, df_afc, df_link=pd.DataFrame(model_rows),pd.DataFrame(afc_rows),pd.DataFrame(linkage_rows)
    # save python dataframe :) and plot later :)
    pkl.dump([df_model,df_afc,df_link],open(plot_path+'LD_spearman_corr.pkl','wb'))
    print('saved all linkage correlations :)')






if __name__ == '__main__':
    plot_path = 'plots/compare_all_abs'
    work_idx=0

    wf_file_end = 'no_n_new_s_pos.pkl'
    use_mean = False
    work_from = 'mogon'
    tmp_data_folder = 'data/'
    num_submodels = 3

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    plot_path += '/'


    finetuning = True
    use_only_test = True

    time_stamp = time.time()
    KDE_bandwidth = 0.05
    summary_method = [0, 1, 2][1]


    noise_paths = ["0", "4", "8", "12"]
    num_targets =["10", "25", "50"]
    sampling_noise = ["", "noise14"]
    coverage = [1000, 40]
    Nsampling =[1000,100]
    use_gene = ['no w', 'w'] #
    use_gene_str =  ['0', '1'] #
    use_gene_end=['_','_abs_']
    num_submodels=3

    model_names_begin=['VVAE_max_']
    model_names_end=['_']
    plot_name=['VAE']
    batch_size=[6]
    KLD=[5]

    input_data=[]
    # Loop over parameters
    for noise_path in noise_paths:
        for i, (cov, sampling, sampling_str) in enumerate(zip(coverage, Nsampling, sampling_noise)):
            for num_target in num_targets:
                animal = f"max_{noise_path}_r_t_{num_target}_n_{sampling_str}"
                animal_no_noise=f"max_{noise_path}_r_t_{num_target}_n_"

                for m_names_begin,m_names_end,plot_n,b_size,KL in zip(model_names_begin,model_names_end,plot_name,batch_size,KLD):
                    nn_name = f"{m_names_begin}{noise_path}_r_t_{num_target}_n{sampling_str}_usegene1{m_names_end}KLD"+str(KL)
                    input_data.append([nn_name,plot_n,b_size,True,animal,noise_path,animal_no_noise,'('+str(sampling)+','+str(cov)+')',num_target])



    sns.set_style("whitegrid")
    debugg=1
    chrom='2L'
    print('Third plot finished')
    make_sum_paper_plot_no_wf(tmp_data_folder, input_data, num_submodels, finetuning, plot_path=plot_path,
                              chrom='2L')

    for linkage_def_idx in [1]:
        print(linkage_def_idx)
        linkage_correlation_filter(input_data,linkage_def_idx,num_neigbors=50,
                            plot_path=plot_path+str(linkage_def_idx),num_submodels=num_submodels,chrom=chrom,debugg=debugg)

