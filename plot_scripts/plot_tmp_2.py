import pickle as pkl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from variables import fix_variables
from scipy import stats
import sys
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
import time
import tensorflow as tf
from evaluation_metrics import all_metrics, all_metrics_list
import matplotlib
import os
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


def load_model_data_noise_less_info(entry, summary_method=1, chrom='2L', work_from='mogon', num_submodels=3,
                                    tmp_data_folder='', target_knowledge=True, identity_gen=5):
    # model_name, plot_name, time_bz, _, animal_noise, ds_name, animal=entry
    (model_name, plot_name, time_bz, finetuning, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
    eval_path = fix_variables[work_from][animal]['eval_positions']
    print(eval_path)
    linkage_data = pkl.load(open(eval_path, 'rb'))
    if target_knowledge:
        num_targets = int(len(linkage_data['selected']) / 100)
        print('--------num targets VAE', num_targets)
        real_selected_points = [linkage_data['selected'][50 + (s_idx * 100)] for s_idx in range(num_targets)]

        idx_lists = [linkage_data['unselected'], real_selected_points]  # [[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data = [
            list(range(len(linkage_data['selected']), len(linkage_data['selected']) + len(linkage_data['unselected']))),
            [list(range(len(linkage_data['selected'])))[50 + (s_idx * 100)] for s_idx in range(num_targets)]]
    else:
        idx_lists = [linkage_data['unselected'][chrom], linkage_data['selected'][chrom]]  # [[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data = [
            list(range(len(linkage_data['selected'][chrom]),
                       len(linkage_data['selected'][chrom]) + len(linkage_data['unselected'][chrom]))),
            list(range(len(linkage_data['selected'][chrom])))]

    noise_freqs = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        noise_freqs = np.squeeze(noise_freqs[chrom])
    print(work_from, animal_noise, chrom, noise_freqs.shape, flush=True)
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
        if chrom != '':
            prediction = prediction[chrom]
        print(prediction.shape)

        std_pred = np.std(prediction, axis=1)
        mean_pred = np.mean(prediction, axis=1)


        prediction = np.transpose(prediction, (0, 2, 1))
        gt_freq_replicated = np.repeat(np.expand_dims(gt_freq, axis=2), repeats=10, axis=2)
        gt_freq_replicated = np.reshape(gt_freq_replicated,
                                        (gt_freq_replicated.shape[0], gt_freq_replicated.shape[1],
                                         gt_freq_replicated.shape[2] * gt_freq_replicated.shape[3]))

        remaining_results = {
            'std': [np.mean(np.abs(std_gt[time_bz - 1:, idx_lists[0]] - std_pred[:, idx_list_model_data[0]]), axis=1),
                    np.mean(np.abs(std_gt[time_bz - 1:, idx_lists[1]] - std_pred[:, idx_list_model_data[1]]), axis=1)],

            'mean': [
                np.mean(np.abs(mean_gt[time_bz - 1:, idx_lists[0]] - mean_pred[:, idx_list_model_data[0]]), axis=1),
                np.mean(np.abs(mean_gt[time_bz - 1:, idx_lists[1]] - mean_pred[:, idx_list_model_data[1]]), axis=1)],

            'MAE': [
                np.mean(np.abs(gt_freq_replicated[time_bz - 1:, idx_lists[0]] - prediction[:, idx_list_model_data[0]]),
                        axis=(1, 2)),  # gt_freq (generation?,snps,reps)
                np.mean(np.abs(gt_freq_replicated[time_bz - 1:, idx_lists[1]] - prediction[:, idx_list_model_data[1]]),
                        axis=(1, 2))],
            'RMSE': [np.mean(
                np.abs(gt_freq_replicated[time_bz - 1:, idx_lists[0]] - prediction[:, idx_list_model_data[0]]) ** 2,
                axis=(1, 2)),  # gt_freq (generation?,snps,reps)
                np.mean(np.abs(
                    gt_freq_replicated[time_bz - 1:, idx_lists[1]] - prediction[:, idx_list_model_data[1]]) ** 2,
                        axis=(1, 2))]

        }
        for metric in all_metrics_list:
            remaining_results[metric.internal_name] = []

        for idx_list_gt, idx_list_model in zip(idx_lists, idx_list_model_data):
            for metric in all_metrics_list:
                remaining_results[metric.internal_name].append(np.zeros((prediction.shape[0])))
            for gen in range(prediction.shape[0]):
                for metric in all_metrics[3]:
                    remaining_results[metric.internal_name][-1][gen] = metric.eval_function(
                        gt_freq[gen + (time_bz - 1), idx_list_gt], prediction[gen, idx_list_model],
                        noise_freqs[idx_list_gt],
                        summary_method)
        in_stream.close()
        plot_data.append(remaining_results)
    return plot_data


def read_wf_file_noise_less_info(animal_noise, animal, summary_method=1, chrom='2L', tmp_data_folder='', wf_file_end='',
                                 work_from='', target_knowledge=True, identity_gen=5):
    num_train_it = 0
    animal_noise_ = animal_noise.replace('_flip', '')
    wf_file_name = tmp_data_folder + 'wf_data/wf_simulations_' + animal_noise_ + '_' + str(num_train_it) + wf_file_end
    print('WF file name', wf_file_name)
    in_stream = open(wf_file_name, 'rb')
    wf_data = pkl.load(in_stream)
    in_stream.close()
    gt_data = dict()
    ###################################
    # was data flipped? then flipp also WF results to match the dataset:
    flips = []
    if 'flip' in animal_noise:
        # read flip positions
        sync_file = fix_variables[work_from][animal]["sync_process"]
        data = pkl.load(open(sync_file, 'rb'))
        print(animal, data.keys(), flush=True)
        if 'flips' in data:
            flips = data['flips'][chrom]
        else:
            eval_path = fix_variables[work_from][animal]['eval_positions']
            idx_list_dict = pkl.load(open(eval_path, 'rb'))
            idx_list = np.asarray(idx_list_dict['selected'][chrom]).tolist() + np.asarray(
                idx_list_dict['unselected'][chrom]).tolist()
            nucleotide_data_flip = data['nucleotide_idx'][chrom]
            sync_file = fix_variables[work_from][animal_noise_]["sync_process"]
            data = pkl.load(open(sync_file, 'rb'))
            nucleotide_data_org = data['nucleotide_idx'][chrom]
            flips = []
            for i_dx, i in enumerate(idx_list):
                if nucleotide_data_org[i][0] != nucleotide_data_flip[i][0]:
                    flips.append(i_dx)
            print('FLIP analysis', nucleotide_data_org.shape[0], len(flips), flush=True)

    ####################################################################################################
    eval_path = fix_variables[work_from][animal]['eval_positions']
    print(eval_path)
    linkage_data = pkl.load(open(eval_path, 'rb'))
    if target_knowledge:
        num_targets = int(len(linkage_data['selected']) / 100)
        print('--------num targets', num_targets)
        real_selected_points = [linkage_data['selected'][50 + (s_idx * 100)] for s_idx in range(num_targets)]
        idx_lists = [linkage_data['unselected'], real_selected_points]  # [[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data = [
            list(range(len(linkage_data['selected']), len(linkage_data['selected']) + len(linkage_data['unselected']))),
            [list(range(len(linkage_data['selected'])))[50 + (s_idx * 100)] for s_idx in range(num_targets)]]
        print('selected positions', real_selected_points)
    else:
        idx_lists = [linkage_data['unselected'][chrom], linkage_data['selected'][chrom]]  # [[0,1,2,4],[0,1,2,3,4]]#
        idx_list_model_data = [list(range(len(linkage_data['selected'][chrom]),
                                          len(linkage_data['selected'][chrom]) + len(
                                              linkage_data['unselected'][chrom]))),
                               list(range(len(linkage_data['selected'][chrom])))]
        print(len(idx_lists[0]), len(idx_lists[1]))

    ####################################################################################################
    noise_freqs = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))['alleles']
    if chrom != '':
        noise_freqs = np.squeeze(noise_freqs[chrom])
    noise_freqs = np.transpose(noise_freqs, (2, 0, 1))[1:]
    gt_stuff = pkl.load(open(fix_variables[work_from][animal]['sync_process'], 'rb'))
    gt_freqs = gt_stuff['alleles']
    position_data = gt_stuff['distances']

    if chrom != '':
        gt_freqs = np.squeeze(gt_freqs[chrom])
        position_data = np.squeeze(position_data[chrom])
    # print('selected positions', position_data[real_selected_points])
    gt_freq = np.transpose(gt_freqs, (2, 0, 1))[1:]

    mean_gt = np.mean(gt_freq, axis=2)
    std_gt = np.std(gt_freq, axis=2)
    noise_freqs = noise_freqs[identity_gen]
    prediction = wf_data['freqs'][chrom]
    print(prediction.shape, flush=True)
    prediction[:, :, :, flips] = 1 - prediction[:, :, :, flips]

    std_pred = np.std(prediction, axis=(0, 2))[1:]

    mean_pred = np.mean(prediction, axis=(0, 2))[1:]
    prediction = np.transpose(prediction, (1, 3, 0, 2))
    prediction = np.reshape(prediction,
                            (prediction.shape[0], prediction.shape[1], prediction.shape[2] * prediction.shape[3]))
    # TODO look if rashape is right

    print('DEBUG shapes of prediction:', prediction.shape, 'gt freq shape',
          gt_freq.shape, 'std_gt.shape', std_gt.shape, 'std_pred.shape', std_pred.shape)

    gt_freq_replicated = np.repeat(np.expand_dims(gt_freq, axis=-1), repeats=10, axis=-1)
    gt_freq_replicated = np.reshape(gt_freq_replicated,
                                    (gt_freq_replicated.shape[0], gt_freq_replicated.shape[1],
                                     gt_freq_replicated.shape[2] * gt_freq_replicated.shape[3]))

    prediction = prediction[1:]

    debug_selected_mean = np.mean(prediction[:, idx_list_model_data[1][:2]], axis=-1)
    print('example points')
    for i_d in range(2):
        for gen in range(debug_selected_mean.shape[0]):
            print(np.round(debug_selected_mean[gen, i_d], 2), end='\t')
        print()

    remaining_results = {
        'std': [np.mean(np.abs(std_gt[:, idx_lists[0]] - std_pred[:, idx_list_model_data[0]]), axis=1),
                np.mean(np.abs(std_gt[:, idx_lists[1]] - std_pred[:, idx_list_model_data[1]]), axis=1)],

        'mean': [np.mean(np.abs(mean_gt[:, idx_lists[0]] - mean_pred[:, idx_list_model_data[0]]), axis=1),
                 np.mean(np.abs(mean_gt[:, idx_lists[1]] - mean_pred[:, idx_list_model_data[1]]), axis=1)],

        'MAE': [
            np.mean(np.abs(gt_freq_replicated[:, idx_lists[0]] - prediction[:, idx_list_model_data[0]]), axis=(1, 2)),
            # gt_freq (generation?,snps,reps)
            np.mean(np.abs(gt_freq_replicated[:, idx_lists[1]] - prediction[:, idx_list_model_data[1]]), axis=(1, 2))],
        'RMSE': [np.mean(np.abs(gt_freq_replicated[:, idx_lists[0]] - prediction[:, idx_list_model_data[0]]) ** 2,
                         axis=(1, 2)),  # gt_freq (generation?,snps,reps)
                 np.mean(np.abs(gt_freq_replicated[:, idx_lists[1]] - prediction[:, idx_list_model_data[1]]) ** 2,
                         axis=(1, 2))]

    }

    noise_freqs_repeated = np.repeat(np.expand_dims(noise_freqs, axis=0), axis=0, repeats=gt_freq.shape[0])
    noise_freqs_repeated_std = np.std(noise_freqs_repeated, axis=2)
    noise_freqs_repeated_mean = np.mean(noise_freqs_repeated, axis=2)

    identity_result = {
        'std': [np.mean(np.abs(std_gt[:, idx_lists[0]] - noise_freqs_repeated_std[:, idx_lists[0]]), axis=1),
                np.mean(np.abs(std_gt[:, idx_lists[1]] - noise_freqs_repeated_std[:, idx_lists[1]]), axis=1)],

        'mean': [np.mean(np.abs(mean_gt[:, idx_lists[0]] - noise_freqs_repeated_mean[:, idx_lists[0]]), axis=1),
                 np.mean(np.abs(mean_gt[:, idx_lists[1]] - noise_freqs_repeated_mean[:, idx_lists[1]]), axis=1)],

        'MAE': [np.mean(np.abs(gt_freq[:, idx_lists[0]] - noise_freqs_repeated[:, idx_lists[0]]), axis=(1, 2)),
                # gt_freq (generation?,snps,reps)
                np.mean(np.abs(gt_freq[:, idx_lists[1]] - noise_freqs_repeated[:, idx_lists[1]]), axis=(1, 2))],
        'RMSE': [np.mean(np.abs(gt_freq[:, idx_lists[0]] - noise_freqs_repeated[:, idx_lists[0]]) ** 2, axis=(1, 2)),
                 # gt_freq (generation?,snps,reps)
                 np.mean(np.abs(gt_freq[:, idx_lists[1]] - noise_freqs_repeated[:, idx_lists[1]]) ** 2, axis=(1, 2))]

    }

    # ,TODO 'MAE','RMSE'

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

    return remaining_results, identity_result


def make_sum_paper_plot_no_wf(tmp_data_folder, input_data, num_submodels,
                              chrom='2L', max_time_bz=6, num_generations=16,
                              cut_time=1, wf_file_end='', work_from='mogon', df=None, columns=[],
                              target_knowledge=True, generations_separate=False, identity_gen=5, ti=0):
    all_noise_ds = dict()
    for entry in input_data:
        (model_name, plot_name, time_bz, fine_tune, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
        all_noise_ds[animal_noise + '\t' + animal] = [ds_name, dataset_type, p_LD_influence]
    if wf_file_end != '':
        # if wf data exists:
        for key, values in all_noise_ds.items():
            animal_noise, animal = key.split('\t')
            ds_name, dataset_type, p_LD_influence = values
            # now calculate wf stuff :)
            print([ds_name, dataset_type, p_LD_influence])
            plot_data, identity_results = read_wf_file_noise_less_info(animal_noise, animal, summary_method=1,
                                                                       chrom=chrom,
                                                                       tmp_data_folder=tmp_data_folder,
                                                                       wf_file_end=wf_file_end,
                                                                       work_from=work_from,
                                                                       target_knowledge=target_knowledge,
                                                                       identity_gen=identity_gen)
            for metric_name in ['relative_mean_error', 'relative_std_error', 'mean', 'std', 'MAE', 'RMSE']:
                for snp_idx, snp_type in enumerate(['unselected', 'selected']):
                    if metric_name in ['MAE'] and snp_idx == 1:
                        print('WF', np.mean(plot_data[metric_name][snp_idx][max_time_bz + cut_time - 1:num_generations],
                                            axis=0),
                              plot_data[metric_name][snp_idx][max_time_bz + cut_time - 1: num_generations])
                        print('id', np.mean(
                            identity_results[metric_name][snp_idx][max_time_bz + cut_time - 1:num_generations], axis=0),
                              identity_results[metric_name][snp_idx][max_time_bz + cut_time - 1: num_generations])

                    if generations_separate:
                        test_gens_result = plot_data[metric_name][snp_idx][max_time_bz + cut_time - 1:num_generations]
                        for test_gen, entry in enumerate(test_gens_result):
                            meta_data = (
                                ds_name, snp_type, 0, entry, metric_name, dataset_type, p_LD_influence, 'WF model',
                                chrom,
                                test_gen)
                            row = {columns[i]: meta_data[i] for i in range(len(columns))}
                            # Append the dictionary as a new row to the DataFrame
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        # save results of identity baseline:
                        if metric_name in ['mean', 'std', 'MAE', 'RMSE']:
                            test_gens_result = identity_results[metric_name][snp_idx][
                                               max_time_bz + cut_time - 1:num_generations]
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
                        meta_data = (
                            ds_name, snp_type, 0, result_data_gens, metric_name, dataset_type, p_LD_influence,
                            'WF model',
                            chrom)
                        row = {columns[i]: meta_data[i] for i in range(len(columns))}
                        # Append the dictionary as a new row to the DataFrame
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

                        # save results of identity baseline:
                        if metric_name in ['mean', 'std', 'MAE', 'RMSE']:
                            result_data_gens = np.mean(
                                identity_results[metric_name][snp_idx][max_time_bz + cut_time - 1:num_generations],
                                axis=0)
                            meta_data = (
                                ds_name, snp_type, 0, result_data_gens, metric_name, dataset_type, p_LD_influence, 'ID',
                                chrom)
                            row = {columns[i]: meta_data[i] for i in range(len(columns))}
                            # Append the dictionary as a new row to the DataFrame
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # TODO for ,'MAE','RMSE' baseline

    for ax_i, entry in enumerate(input_data):

        print(ax_i, 'load model', entry, flush=True)
        plot_data = load_model_data_noise_less_info(entry, summary_method=1, chrom=chrom,
                                                    work_from=work_from, num_submodels=num_submodels,
                                                    tmp_data_folder=tmp_data_folder, target_knowledge=target_knowledge,
                                                    identity_gen=identity_gen)
        # plot_data=np.asarray(plot_data)
        # plot_data_ds.append(plot_data)
        # (model_name, plot_name, time_bz, finetuning, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
        tbs = entry[2]
        train_gens = fix_variables[entry[4]]['num_train_generations'][ti]
        cut_time = train_gens - tbs
        print(entry[-2], end='\t')
        for submodel_num in range(num_submodels):
            submodel_data = plot_data[submodel_num]
            for metric_name in ['relative_mean_error', 'relative_std_error', 'mean', 'std', 'MAE', 'RMSE']:
                for snp_idx, snp_type in enumerate(['unselected', 'selected']):
                    if metric_name == 'relative_mean_error' and snp_idx == 1:
                        print('VAE', np.mean(submodel_data[metric_name][snp_idx][cut_time:num_generations], axis=0),
                              metric_name, submodel_data[metric_name][snp_idx][cut_time:num_generations]
                              )

                    if generations_separate:
                        test_gens_result = submodel_data[metric_name][snp_idx][cut_time:num_generations]
                        print(test_gens_result)
                        print(test_gens_result.shape)
                        for test_gen, entry_val in enumerate(test_gens_result):
                            meta_data = (
                                entry[-4], snp_type, submodel_num, entry_val, metric_name, entry[-2], entry[-1],
                                entry[1],
                                chrom, test_gen)
                            row = {columns[i]: meta_data[i] for i in range(len(columns))}
                            # Append the dictionary as a new row to the DataFrame
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    else:
                        result_data_gens = np.mean(submodel_data[metric_name][snp_idx][cut_time:num_generations],
                                                   axis=0)
                        meta_data = (
                            entry[-4], snp_type, submodel_num, result_data_gens, metric_name, entry[-2], entry[-1],
                            entry[1], chrom)
                        row = {columns[i]: meta_data[i] for i in range(len(columns))}
                        # Append the dictionary as a new row to the DataFrame
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def calculate_naive_sim_scores(gt_freqs, idx_list_pkl_data, use_abs=True, num_train_gens=6):
    gt_freqs = np.transpose(gt_freqs, (1, 0, 2))  # (2,0,1))
    print(len(idx_list_pkl_data))
    print(gt_freqs.shape, flush=True)
    naive_sim_scores = []

    valid_indices = []
    neighbors = []
    valid_indices_2 = []
    # this calculates the afc and probably makes the baseline too good
    # gt_freqs = gt_freqs - gt_freqs[:, 0:1]

    for i, idx in enumerate(idx_list_pkl_data):
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

    # print('neighbors.shape', neighbors.shape, neighbors[0])
    for replicate in range(10):
        focal_SNP_rep = np.expand_dims(focal_snp[:, :, replicate], axis=1)  # shape (1000,1,6)
        neighbors_rep = neighbors[:, :, :, replicate]  # shape (1000,50,6)

        focus_enc2_normalized = tf.nn.l2_normalize(focal_SNP_rep, axis=-1)  # Shape: (batch_size, 1, latent_dim)
        neighbor_enc2_normalized = tf.nn.l2_normalize(neighbors_rep, axis=-1)  # Shape: (batch_size, n, latent_dim)

        similarity_scores = tf.matmul(focus_enc2_normalized, neighbor_enc2_normalized, transpose_b=True).numpy()
        similarity_scores = np.squeeze(similarity_scores)
        print(similarity_scores.shape, 'shape')
        flatten_sim_scores = []
        for entry in similarity_scores:
            flatten_sim_scores.extend(entry.tolist())

        naive_sim_scores.append(flatten_sim_scores)
    naive_sim_scores = np.asarray(naive_sim_scores)
    if use_abs:
        naive_sim_scores = np.abs(naive_sim_scores)
    # naive_sim_scores = np.mean(naive_sim_scores, axis=0)
    naive_sim_scores = np.transpose(naive_sim_scores, (1, 0))

    print('flatten_sim_scores', naive_sim_scores.shape, flush=True)
    return naive_sim_scores, valid_indices, valid_indices_2





def calculate_corr_values(filtered_actual_linkage, filtered_pot_linkage):
    filtered_actual_linkage = np.asarray(filtered_actual_linkage, dtype=float)
    filtered_pot_linkage = np.asarray(filtered_pot_linkage, dtype=float)
    mae_model = np.mean(np.abs(filtered_actual_linkage - filtered_pot_linkage))
    rmse_model = np.sqrt(np.mean((filtered_actual_linkage - filtered_pot_linkage) ** 2))
    spear = stats.spearmanr(filtered_actual_linkage, filtered_pot_linkage)
    pears = stats.pearsonr(filtered_actual_linkage, filtered_pot_linkage)
    xi = 0
    return spear[0], pears[0], mae_model, rmse_model, xi




def read_LDx(file_paths, filter_name='.txt', ld_col=12):
    """Return {canonical_pair: (mean_LD, mean_joint_depth)}.

    ld_col is zero-based: 12 preserves the approximate MLE used here;
    14 selects the direct r-squared estimate. Column 11 (index 10) is
    joint read depth. Both means use the same available LDx records.
    """
    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]
    snp_dict = {}
    for file_path in file_paths:
        file_path = os.fspath(file_path)
        if file_path.endswith('.txt'):
            file_path = file_path[:-4]
        file_path += filter_name
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                parts = line.rstrip('\n').split('\t')
                try:
                    snp1, snp2 = sorted((int(parts[0]), int(parts[1])))
                    estimate = float(parts[ld_col])
                except (IndexError, ValueError) as exc:
                    print(f'Skipping {file_path}:{line_num}: {exc}')
                    continue
                if not np.isfinite(estimate):
                    print(f'Skipping {file_path}:{line_num}: non-finite LD estimate')
                    continue

                # Missing quality must not remove a valid estimate at level 0.
                try:
                    joint_depth = float(parts[10])
                except (IndexError, ValueError):
                    joint_depth = np.nan
                if not np.isfinite(joint_depth) or joint_depth < 0:
                    joint_depth = np.nan
                    print(f'Missing/invalid joint depth at {file_path}:{line_num}')

                key = f'{snp1}_{snp2}'
                sums = snp_dict.setdefault(key, [0.0, 0.0, 0])
                sums[0] += estimate  # A zero LD estimate is valid.
                sums[1] += joint_depth
                sums[2] += 1
        print('#### LDx entries:', len(snp_dict))

    # If a contributing record lacks depth, keep its LD mean but leave its
    # quality unknown (NaN); only levels 1..19 require known joint depth.
    return {key: (ld_sum / n, depth_sum / n)
            for key, (ld_sum, depth_sum, n) in snp_dict.items()}


base_to_idx = {'a': 0, 'A': 0, 't': 1, 'T': 1, 'c': 2, 'C': 2, 'g': 3, 'G': 3}


def read_LDx_r(file_paths, nucleotide_idxs, idx_focus, idx_neighbors,
               distances, filter_name='.txt'):
    """Return {canonical_pair: (mean_LD, mean_joint_depth)}.
    calculate r instead of r^2 based on read counts
    """
    # gt_order=set(zip(idx_focus,idx_neighbors))

    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]

    # Fast lookup instead of np.where(distances == snp)[0][0]
    distance_to_idx = {int(pos): i for i, pos in enumerate(distances)}

    snp_dict = {}

    for file_path in file_paths:
        file_path = os.fspath(file_path)

        if file_path.endswith('.txt'):
            file_path = file_path[:-4]

        file_path += filter_name

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):

                if not line.strip() or line.lstrip().startswith('#'):
                    continue

                parts = line.rstrip('\n').split('\t')

                try:
                    snp1, snp2 = sorted((int(parts[0]), int(parts[1])))

                    snp_nucleotide_1 = base_to_idx[parts[15]]
                    snp_nucleotide_2 = base_to_idx[parts[17]]

                    arr_idx_1 = distance_to_idx[snp1]
                    arr_idx_2 = distance_to_idx[snp2]

                    nucleotides1 = nucleotide_idxs[arr_idx_1][0]
                    nucleotides2 = nucleotide_idxs[arr_idx_2][0]

                    AB, aB, Ab, ab = map(float, parts[2:6])

                    denominator = np.sqrt(
                        (AB + Ab) *
                        (aB + ab) *
                        (AB + aB) *
                        (Ab + ab)
                    )

                    # Bei monomorphen Positionen ist r nicht definiert.
                    if denominator > 0:
                        estimate = (AB * ab - Ab * aB) / denominator

                        # if (snp1,snp2) in gt_order: # no order change
                        #     print('inside')

                        if (
                                (snp_nucleotide_1 == nucleotides1 and
                                 snp_nucleotide_2 != nucleotides2)
                                or
                                (snp_nucleotide_2 == nucleotides2 and
                                 snp_nucleotide_1 != nucleotides1)
                        ):
                            estimate = -estimate

                    else:
                        continue

                except (IndexError, ValueError, KeyError) as exc:
                    # print(f'Skipping {file_path}:{line_num}: {exc}')
                    continue

                if not np.isfinite(estimate):
                    print(
                        f'Skipping {file_path}:{line_num}: '
                        f'non-finite LD estimate'
                    )
                    continue

                # Missing quality must not remove a valid estimate at level 0.
                try:
                    joint_depth = float(parts[10])
                except (IndexError, ValueError):
                    joint_depth = np.nan

                if not np.isfinite(joint_depth) or joint_depth < 0:
                    joint_depth = np.nan
                    print(
                        f'Missing/invalid joint depth at '
                        f'{file_path}:{line_num}'
                    )

                key = f'{snp1}_{snp2}'
                sums = snp_dict.setdefault(key, [0.0, 0.0, 0])

                sums[0] += estimate  # A zero LD estimate is valid.
                sums[1] += joint_depth
                sums[2] += 1

        print('#### LDx entries:', len(snp_dict))

    # If a contributing record lacks depth, keep its LD mean but leave its
    # quality unknown (NaN); only levels 1..19 require known joint depth.
    return {
        key: (ld_sum / n, depth_sum / n)
        for key, (ld_sum, depth_sum, n) in snp_dict.items()
    }





def get_LDx_intersections(val_indices1, val_indices2, LDx_str_ps, distances):
    """Align LD and joint depth with the existing +/-50-neighbour order."""
    if len(val_indices1) != len(val_indices2):
        raise ValueError('val_indices1 and val_indices2 must have equal length')
    LDx_result_flat = []
    LDx_joint_depth = []
    for i, idx in zip(val_indices1, val_indices2):
        if idx < 50 or idx + 51 > len(distances):
            raise ValueError(f'Focus index {idx} lacks 50 neighbours on each side')
        p0 = int(distances[idx])
        for p1 in np.concatenate([distances[idx - 50:idx],
                                  distances[idx + 1:idx + 51]]):
            p1 = int(p1)
            key = f'{min(p0, p1)}_{max(p0, p1)}'
            estimate, joint_depth = LDx_str_ps.get(key, (np.nan, np.nan))
            LDx_result_flat.append(estimate)
            LDx_joint_depth.append(joint_depth)

    LDx_result_flat = np.asarray(LDx_result_flat, dtype=float)
    LDx_joint_depth = np.asarray(LDx_joint_depth, dtype=float)
    LDx_result_mask = np.isfinite(LDx_result_flat)
    print('LDx pairs total/found:', len(LDx_result_flat),
          np.count_nonzero(LDx_result_mask))
    return LDx_result_flat, LDx_result_mask, LDx_joint_depth


def summarize_values(vals_pos, mask_pos, vals_neg, mask_neg):
    num_pos = np.count_nonzero(mask_pos)
    num_neg = np.count_nonzero(mask_neg)
    return (num_pos * vals_pos + vals_neg * num_neg) / (num_neg + num_pos)
    # return (vals_pos + vals_neg) / 2


def calculate_LDcorrelation(pot_linkage, linkage_trust_arr, linkage_trust_dict,
                            afc_joint_reps, actual_linkage, afc_focus, afc_neighbor, afc_weight_sum, data,
                            chrom, val_indices2, idx_neighbors, naive_sim_scores, model_rows, ds_name, dataset_type,
                            p_LD_influence,
                            submodel_num, model_name, afc_rows, plot_path, linkage_idx, plot_name, r2_mask_pos,
                            r2_mask_neg,
                            idx_focus, gt_freqs, afc_gen, LDx_masks,
                            filter_modes=['coherence', 'afc_min', 'afc_geom', 'afc_max']):
    aggregation = ['mean', 'afc_weight', 'afc_weight_middle', ''][0]
    pot_linkage = pot_linkage.reshape(pot_linkage.shape[0], -1, 10).mean(axis=1)

    x = np.asarray(pot_linkage)  # (N_Paare, 10)
    # '''
    abs_sum = np.abs(x).sum(axis=1)
    coherence = np.divide(
        np.abs(x.sum(axis=1)),
        abs_sum,
        out=np.zeros(x.shape[0], dtype=float),
        where=abs_sum > 0,
    )


    x_sim = np.asarray(naive_sim_scores)  # (N_Paare, 10)
    # '''
    abs_sum_sim = np.abs(x_sim).sum(axis=1)
    coherence_sim = np.divide(
        np.abs(x_sim.sum(axis=1)),
        abs_sum_sim,
        out=np.zeros(x_sim.shape[0], dtype=float),
        where=abs_sum_sim > 0,
    )

    if aggregation == 'afc_weight':
        print("afc_weight")
        pot_linkage = np.divide(
            np.einsum('ij,ij->i', pot_linkage, afc_joint_reps),
            afc_weight_sum,
            out=pot_linkage.mean(axis=1),
            where=afc_weight_sum > 0
        )
    elif aggregation == "mean":
        print("mean")
        # pot_linkage = pot_linkage.mean(axis=1)
        pot_linkage = np.mean(pot_linkage, axis=1)
        naive_sim_scores = np.mean(naive_sim_scores, axis=1)
    elif aggregation == 'afc_weight_middle':
        print("afc_weight_middle")
        w = np.sqrt(afc_focus * afc_neighbor)
        w_sum = w.sum(axis=1)
        pot_linkage = np.divide(
            (w * x).sum(axis=1), w_sum,
            out=x.mean(axis=1), where=w_sum > 0
        )
    else:
        print("weight_middle")
        u = np.mean(x * np.abs(x), axis=1)
        pot_linkage = np.sign(u) * np.sqrt(np.abs(u))

    # if not dataset_type=='(100,40)':
    if 'all_z_tide' in data:
        print('all_z_tide')
        summarized_embedding = data['all_z_tide']
        # print('summarized_embedding',summarized_embedding)
        if isinstance(summarized_embedding, dict):
            summarized_embedding = summarized_embedding[chrom]
        # summarized_embedding=summarized_embedding[]
        summarized_embedding = np.asarray(summarized_embedding)
        summarized_embedding = np.squeeze(summarized_embedding)
        print(summarized_embedding.shape)

        embedding_focus = np.repeat(summarized_embedding[np.asarray(val_indices2), :], repeats=100,
                                    axis=0)  # (num_nps,sample_size)
        embedding_neighbor = summarized_embedding[idx_neighbors]  # (num_nps,sample_size)
        print(embedding_focus.shape)
        embedding_focus_mean = np.mean(embedding_focus, axis=-1)
        embedding_neighbor_mean = np.mean(embedding_neighbor, axis=-1)
        mask_zeros = (embedding_neighbor_mean == 0) | (embedding_focus_mean == 0)
        print('Count zero embeddings:', np.count_nonzero(mask_zeros), mask_zeros.shape)
        mask_zeros = ~mask_zeros
    else:
        print('all_z_tide NOT IN DATA !!!!!!!!!!!')

    zeros_of_pot_linkage = (pot_linkage == 0)
    print(np.count_nonzero(zeros_of_pot_linkage))
    zeros_of_pot_linkage = ~zeros_of_pot_linkage
    print(np.count_nonzero(zeros_of_pot_linkage))

    if len(linkage_trust_dict):
        base_mask = (linkage_trust_arr == 1.0) & (zeros_of_pot_linkage)
    else:
        base_mask = (r2_mask_neg | r2_mask_pos) & (zeros_of_pot_linkage)
    quantile_levels = np.linspace(0.0, 0.90, 20)
    coherence_thresholds = np.quantile(
        coherence[base_mask], quantile_levels
    )
    coherence_thresholds[0] = 0.0
    coherence_thresholds_sim = np.quantile(
        coherence_sim[base_mask], quantile_levels
    )
    coherence_thresholds_sim[0] = 0.0

    #######################################
    print(filter_modes)
    for filter_name in filter_modes:

        if filter_name == 'none':
            quality = np.ones(len(actual_linkage))
            quality_sim = np.ones(len(actual_linkage))
        elif filter_name == 'coherence':
            quality = coherence
            quality_sim = coherence_sim
        elif filter_name == 'afc_min':
            quality = np.minimum(
                afc_focus.mean(axis=1), afc_neighbor.mean(axis=1)
            )
            quality_sim = quality

        elif filter_name == 'afc_max':
            quality = np.maximum(
                afc_focus.mean(axis=1), afc_neighbor.mean(axis=1)
            )
            quality_sim = quality

        elif filter_name == 'afc_geom':
            quality = np.sqrt(
                afc_focus.mean(axis=1) * afc_neighbor.mean(axis=1)
            )
            quality_sim = quality
        elif filter_name == 'afc_sf_min':
            p0 = gt_freqs[0]
            pT = gt_freqs[afc_gen]
            afc_sf = np.mean(
                2.0 * np.abs(np.arcsin(np.sqrt(pT)) - np.arcsin(np.sqrt(p0))),
                axis=1
            )
            quality = np.minimum(afc_sf[idx_focus], afc_sf[idx_neighbors])
            quality_sim = quality
        # else:
        #    raise ValueError('Unknown filter mode: ' + str(filter_name))

        if filter_name == 'none':
            thresholds = [None]
        else:
            thresholds = np.quantile(quality[base_mask], quantile_levels)
            thresholds_sim = np.quantile(quality_sim[base_mask], quantile_levels)

        for filter_idx, threshold in enumerate(thresholds):

            if filter_idx == 0:
                if filter_name == 'LDx_mask':
                    mask = mask_sim = LDx_masks[filter_idx][0]
                else:
                    mask = base_mask.copy()
                    mask_sim = base_mask.copy()
            else:
                if filter_name == 'LDx_mask':
                    mask = mask_sim = LDx_masks[filter_idx][0]
                else:
                    mask = base_mask & (quality >= threshold)
                    mask_sim = base_mask & (quality_sim >= thresholds_sim[filter_idx])

            pos_mask = mask & r2_mask_pos
            neg_mask = mask & r2_mask_neg
            pos_mask_sim = mask_sim & r2_mask_pos
            neg_mask_sim = mask_sim & r2_mask_neg
            ###################################################################

            if 'all_z_tide' in data:
                pos_mask = pos_mask & mask_zeros
                neg_mask = neg_mask & mask_zeros
                pos_mask_sim = pos_mask_sim & mask_zeros
                neg_mask_sim = neg_mask_sim & mask_zeros
            num_true_pos = np.count_nonzero(pos_mask)
            num_true_neg = np.count_nonzero(neg_mask)
            num_true_pos_sim = np.count_nonzero(pos_mask_sim)
            num_true_neg_sim = np.count_nonzero(neg_mask_sim)
            print('num_true_neg,num_true_pos', num_true_neg, num_true_pos)
            if num_true_neg < 30 or num_true_pos < 30:
                break

            correlations = []
            filtered_pot_linkage = pot_linkage[pos_mask]
            filtered_actual_linkage = actual_linkage[pos_mask]
            # small debugging plot
            res_1, pears, mae_model, rmse_model, _ = calculate_corr_values(filtered_actual_linkage,
                                                                           filtered_pot_linkage)
            corr_baseline, pears_base, mae_base, rmse_base, _ = calculate_corr_values(actual_linkage[pos_mask_sim],
                                                                                      naive_sim_scores[pos_mask_sim])

            res_1_neg, pears_neg, mae_model_neg, rmse_model_neg, _ = calculate_corr_values(actual_linkage[neg_mask],
                                                                                           pot_linkage[neg_mask])
            corr_baseline_neg, pears_base_neg, mae_base_neg, rmse_base_neg, _ = calculate_corr_values(
                actual_linkage[neg_mask_sim], naive_sim_scores[neg_mask_sim])

            res_1 = summarize_values(res_1, pos_mask, res_1_neg, neg_mask)
            pears = summarize_values(pears, pos_mask, pears_neg, neg_mask)
            mae_model = summarize_values(mae_model, pos_mask, mae_model_neg, neg_mask)
            rmse_model = summarize_values(rmse_model, pos_mask, rmse_model_neg, neg_mask)

            corr_baseline = summarize_values(corr_baseline, pos_mask_sim, corr_baseline_neg, neg_mask_sim)
            pears_base = summarize_values(pears_base, pos_mask_sim, pears_base_neg, neg_mask_sim)
            mae_base = summarize_values(mae_base, pos_mask_sim, mae_base_neg, neg_mask_sim)
            rmse_base = summarize_values(rmse_base, pos_mask_sim, rmse_base_neg, neg_mask_sim)

            correlations.extend([0, res_1, 0, corr_baseline])

            # save values
            res_1_norm, pears_norm, mae_model_norm, rmse_model_norm, xi_model_norm = calculate_corr_values(
                actual_linkage[mask], pot_linkage[mask])
            corr_baseline_norm, pears_base_norm, mae_base_norm, rmse_base_norm, xi_base_norm = calculate_corr_values(
                actual_linkage[mask_sim], naive_sim_scores[mask_sim])
            afc_diff = filter_idx * 0.01

            model_rows.append({
                'dataset_name': ds_name,
                'dataset_type': dataset_type,
                'p_{LD} influence': p_LD_influence,
                'model name': model_name,
                'model plot name': plot_name,
                'submodel_num': submodel_num, 'AFC filter': afc_diff,
                'LD estimator': 'VAE',
                'Chrom': chrom,
                'Spearman roh diffcomp': res_1,
                'Pearson fair diffcomp': pears,
                'MAE diffcomp': mae_model,
                'RMSE diffcomp': rmse_model,
                'Spearman roh': res_1_norm,
                'Pearson': pears_norm,
                'MAE': mae_model_norm,
                'RMSE': rmse_model_norm,
                'XI': xi_model_norm,
                'num_eval_spns': num_true_pos + num_true_neg,
                'num_all_spns': np.count_nonzero((linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else
                actual_linkage.shape[0],
                'filter': filter_name

            })

            afc_rows.append({
                'dataset_name': ds_name,
                'dataset_type': dataset_type,
                'p_{LD} influence': p_LD_influence,
                'AFC filter': afc_diff,
                'LD estimator': 'org_sim',
                'Chrom': chrom,
                'Spearman roh diffcomp': corr_baseline,
                'Pearson diffcomp': pears_base,
                'MAE diffcomp': mae_base,
                'RMSE diffcomp': rmse_base,
                'Spearman roh': corr_baseline_norm,
                'Pearson': pears_base_norm,
                'MAE': mae_base_norm,
                'RMSE': rmse_base_norm,
                'XI': xi_base_norm,
                'num_eval_spns': num_true_pos_sim + num_true_neg_sim,
                'num_all_spns': np.count_nonzero((linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else
                actual_linkage.shape[0],
                'filter': filter_name

            })
    return model_rows, afc_rows


def calculate_VAE_LDx_regression(pot_linkage, LDx_values, LDx_result_mask, actual_linkage,
                                 gt_freqs, idx_focus, idx_neighbors, distances, afc_gen,
                                 linkage_trust_arr, linkage_trust_dict, r2_mask_pos, r2_mask_neg,
                                 model_rows, ds_name, dataset_type, p_LD_influence,
                                 submodel_num, model_name, plot_name, chrom,
                                 filter_modes=['afc_min'],  # 'none', 'coherence',
                                 train_fraction=0.10, max_train_pairs=5000,
                                 random_state=42, regressor_factory=None, linkage_idx=1,
                                 ):
    # ------------------------------------------------------------------
    # 1. VAE scores: exactly as in calculate_LDcorrelation()
    #    -> one row per SNP pair, 10 replicate scores
    # ------------------------------------------------------------------
    vae_reps = pot_linkage.reshape(pot_linkage.shape[0], -1, 10).mean(axis=1)
    vae_mean = np.mean(vae_reps, axis=1)
    vae_std = np.std(vae_reps, axis=1)

    abs_sum = np.abs(vae_reps).sum(axis=1)
    coherence = np.divide(np.abs(vae_reps.sum(axis=1)), abs_sum,
                          out=np.zeros(vae_reps.shape[0], dtype=float), where=abs_sum > 0)

    # ------------------------------------------------------------------
    # 2. Pairwise SNP information
    # ------------------------------------------------------------------
    idx_focus = np.asarray(idx_focus, dtype=int)
    idx_neighbors = np.asarray(idx_neighbors, dtype=int)

    sf_focus_reps = gt_freqs[0, idx_focus, :]
    sf_neighbor_reps = gt_freqs[0, idx_neighbors, :]

    delta = gt_freqs[afc_gen] - gt_freqs[0]
    delta_focus_reps = delta[idx_focus, :]
    delta_neighbor_reps = delta[idx_neighbors, :]

    sf_focus = np.mean(sf_focus_reps, axis=1)
    sf_neighbor = np.mean(sf_neighbor_reps, axis=1)

    afc_focus = np.mean(np.abs(delta_focus_reps), axis=1)
    afc_neighbor = np.mean(np.abs(delta_neighbor_reps), axis=1)

    X = np.column_stack([
        vae_reps,
        vae_mean,
        vae_std,
        coherence,
        afc_focus,
        afc_neighbor,
        sf_focus,
        sf_neighbor,
        sf_focus * (1.0 - sf_focus),
        sf_neighbor * (1.0 - sf_neighbor),
    ])

    # ------------------------------------------------------------------
    # 4. Base mask
    # ------------------------------------------------------------------
    valid_gt = r2_mask_pos | r2_mask_neg
    zeros_of_pot_linkage = (np.mean(pot_linkage, axis=-1) == 0)
    if len(linkage_trust_dict):
        base_mask = (linkage_trust_arr == 1.0) & valid_gt & (~zeros_of_pot_linkage)
    else:
        base_mask = valid_gt.copy() & (~zeros_of_pot_linkage)

    base_mask &= np.isfinite(actual_linkage)
    finite_features = np.all(np.isfinite(X), axis=1)

    train_candidates = base_mask & LDx_result_mask & np.isfinite(LDx_values) & finite_features
    train_candidates_idx = np.where(train_candidates)[0]

    if len(train_candidates_idx) < 10:
        print('VAE+LDx: too few LDx pairs for regression:', len(train_candidates_idx))
        return model_rows


    pair_keys = np.sort(
        np.column_stack((
            idx_focus[train_candidates_idx],
            idx_neighbors[train_candidates_idx]
        )),
        axis=1
    )
    _, pair_group = np.unique(pair_keys, axis=0, return_inverse=True)

    rng = np.random.default_rng(random_state)
    groups = rng.permutation(pair_group.max() + 1)

    # Fester Split: 80 % Trainingspool, 20 % Holdout.
    split = int(train_fraction * len(groups))
    train_groups = groups[:split][:max_train_pairs]
    holdout_groups = groups[split:]

    train_mask = np.zeros(len(actual_linkage), dtype=bool)
    holdout_mask = np.zeros(len(actual_linkage), dtype=bool)

    train_mask[train_candidates_idx] = np.isin(pair_group, train_groups)
    holdout_mask[train_candidates_idx] = np.isin(pair_group, holdout_groups)

    # Auswertung: LDx-Holdout plus Paare ohne LDx.
    # Nicht verwendete Paare des Trainingspools bleiben ausgeschlossen.
    evaluation_mask = (
            base_mask
            & finite_features
            & (holdout_mask | ~LDx_result_mask)
    )

    print("Training:", np.count_nonzero(train_mask))
    print("LDx-Holdout:", np.count_nonzero(holdout_mask))

    # ------------------------------------------------------------------
    # 6. Train model
    # ------------------------------------------------------------------
    # if regressor_factory is None:
    #    regressor = LinearRegression()
    # else:
    #    regressor = regressor_factory()
    regressor = HistGradientBoostingRegressor(max_depth=3, random_state=42)
    # regressor = make_pipeline( StandardScaler(),Ridge(alpha=1.0))
    regressor.fit(X[train_mask], LDx_values[train_mask])

    pred = np.full(len(actual_linkage), np.nan)
    pred[finite_features] = regressor.predict(X[finite_features])

    pred_raw = pred.copy()
    if linkage_idx == 1:
        pred = np.clip(pred, 0.0, 1.0)
    elif linkage_idx == 3:
        pred = np.clip(pred, -1.0, 1.0)
    print(
        "LDx-Holdout Spearman:",
        stats.spearmanr(
            LDx_values[holdout_mask],
            pred[holdout_mask]
        )[0]
    )

    # DEBUG: random sample from evaluation set BEFORE filtering
    debug_eval_mask = evaluation_mask & finite_features & np.isfinite(pred)
    debug_candidates = np.where(debug_eval_mask)[0]

    rng_debug = np.random.default_rng(random_state + 1)
    debug_idx = rng_debug.choice(debug_candidates, size=min(400, len(debug_candidates)), replace=False)
    np.set_printoptions(threshold=sys.maxsize)
    print('\n######## VAE+LDx DEBUG SAMPLE ########')

    def debug_rho(a, b, mask):
        mask = mask & np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]

        if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return np.nan

        return stats.spearmanr(a, b)[0]

    def debug_fair_rho(a, b, mask):
        mask = mask & np.isfinite(a) & np.isfinite(b)
        pos = mask & r2_mask_pos
        neg = mask & r2_mask_neg

        if np.count_nonzero(pos) < 3 or np.count_nonzero(neg) < 3:
            return np.nan

        return summarize_values(
            debug_rho(a, b, pos), pos,
            debug_rho(a, b, neg), neg
        )


    rng_debug = np.random.default_rng(42)
    positions = np.asarray(distances)

    maf_pair = np.minimum.reduce([
        sf_focus, 1.0 - sf_focus,
        sf_neighbor, 1.0 - sf_neighbor
    ])

    for split_name, split_mask in [
        ("TRAIN", train_mask),
        ("HOLDOUT", holdout_mask),
        ("OHNE_LDx", evaluation_mask & ~LDx_result_mask),
    ]:
        ids = np.flatnonzero(split_mask & np.isfinite(maf_pair))
        if ids.size == 0:
            continue

        # Über die GESAMTE Teilmenge, nicht nur die Stichprobe:
        print(
            f"\n{split_name}: n={ids.size}, "
            f"Anteil MAF_min < 0.10: "
            f"{np.mean(maf_pair[ids] < 0.10):.2%}"
        )

        s = rng_debug.choice(ids, size=min(100, ids.size), replace=False)

        table = pd.DataFrame({
            "pair_idx": s,
            "pos_i": positions[idx_focus[s]],
            "pos_j": positions[idx_neighbors[s]],
            "GT": actual_linkage[s],
            "LDx": LDx_values[s],
            "VAE_raw": vae_mean[s],
            "VAE_LDx": pred[s],
            "SF_i": sf_focus[s],
            "SF_j": sf_neighbor[s],
            "AFC_i": afc_focus[s],
            "AFC_j": afc_neighbor[s],
            "MAF_min": maf_pair[s],
            "coherence": coherence[s],
        })

        print(table.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        ))

    for filter_name in filter_modes:

        if filter_name == 'none':
            quality = np.ones(len(actual_linkage))
        elif filter_name == 'coherence':
            quality = coherence
        elif filter_name == 'afc_min':
            quality = np.minimum(afc_focus, afc_neighbor)
        elif filter_name == 'afc_max':
            quality = np.maximum(afc_focus, afc_neighbor)
        elif filter_name == 'afc_geom':
            quality = np.sqrt(afc_focus * afc_neighbor)
        elif filter_name == 'afc_sf_min':
            p0 = gt_freqs[0]
            pT = gt_freqs[afc_gen]
            afc_sf = np.mean(
                2.0 * np.abs(np.arcsin(np.sqrt(pT)) - np.arcsin(np.sqrt(p0))),
                axis=1
            )
            quality = np.minimum(afc_sf[idx_focus], afc_sf[idx_neighbors])

        else:
            raise ValueError('Unknown filter mode: ' + str(filter_name))

        # thresholds are calculated ONLY from evaluation pairs, never training pairs
        eval_base = evaluation_mask & np.isfinite(pred) & np.isfinite(quality)

        if filter_name == 'none':
            thresholds = [None]
        else:
            thresholds = np.quantile(quality[eval_base], np.linspace(0.0, 0.90, 20))

        for filter_idx, threshold in enumerate(thresholds):

            if filter_idx == 0:
                mask = eval_base.copy()
            else:
                mask = eval_base & (quality >= threshold)

            pos_mask = mask & r2_mask_pos
            neg_mask = mask & r2_mask_neg

            num_true_pos = np.count_nonzero(pos_mask)
            num_true_neg = np.count_nonzero(neg_mask)

            if num_true_pos < 30 or num_true_neg < 30:
                print('VAE+LDx stop filter:', filter_name, filter_idx, num_true_pos, num_true_neg)
                break


            res_pos, pears_pos, mae_pos, rmse_pos, _ = calculate_corr_values(actual_linkage[pos_mask], pred[pos_mask])
            res_neg, pears_neg, mae_neg, rmse_neg, _ = calculate_corr_values(actual_linkage[neg_mask], pred[neg_mask])

            res_1 = summarize_values(res_pos, pos_mask, res_neg, neg_mask)
            pears = summarize_values(pears_pos, pos_mask, pears_neg, neg_mask)
            mae_model = summarize_values(mae_pos, pos_mask, mae_neg, neg_mask)
            rmse_model = summarize_values(rmse_pos, pos_mask, rmse_neg, neg_mask)

            # normal global metrics
            res_1_norm, pears_norm, mae_model_norm, rmse_model_norm, xi = calculate_corr_values(actual_linkage[mask],
                                                                                                pred[mask])

            num_eval = num_true_pos + num_true_neg
            num_without_ldx = np.count_nonzero(mask & ~LDx_result_mask)
            num_with_ldx = np.count_nonzero(mask & LDx_result_mask)

            model_rows.append({
                'dataset_name': ds_name,
                'dataset_type': dataset_type,
                'p_{LD} influence': p_LD_influence,
                'model name': model_name,
                'model plot name': plot_name,
                'submodel_num': submodel_num,
                'AFC filter': filter_idx * 0.01,
                'LD estimator': 'VAE+LDx',
                'Chrom': chrom,
                'Spearman roh diffcomp': res_1,
                'Pearson fair diffcomp': pears,
                'MAE diffcomp': mae_model,
                'RMSE diffcomp': rmse_model,
                'Spearman roh': res_1_norm,
                'Pearson': pears_norm,
                'MAE': mae_model_norm,
                'RMSE': rmse_model_norm,
                'XI': xi,
                'num_eval_spns': num_eval,
                'num_all_spns': np.count_nonzero(linkage_trust_arr == 1.0) if len(linkage_trust_dict) else
                actual_linkage.shape[0],
                'filter': filter_name,
                'train pairs': np.count_nonzero(train_mask),
                'eval pairs with LDx': num_with_ldx,
                'eval pairs without LDx': num_without_ldx
            })

            print('VAE+LDx', filter_name, filter_idx,
                  'train:', np.count_nonzero(train_mask),
                  'eval:', num_eval,
                  'without LDx:', num_without_ldx,
                  'fair Spearman:', res_1,
                  'global Spearman:', res_1_norm)

    return model_rows


def linkage_correlation_filter(model_rows, afc_rows, linkage_rows, base_model_names, linkage_idx, num_neighbors,
                               plot_path, num_submodels=3,
                               chrom='2L', debugg=0, use_abs=True, work_from='mogon', tmp_data_folder='', afc_gen=5,
                               num_train_gens=6, gt_ld_neighbors=50, use_LDx_separate=1, time_series_LD=False,
                               # NEW:
                               run_VAE_LDx=True, vae_ldx_learning_mode='pairwise'):
    sp_idx = 0
    matplotlib.rcParams['lines.markersize'] = 4

    for entry in base_model_names:
        print(entry, flush=True)
        (model_name, plot_name, time_bz, fine_tune, animal_noise, ds_name, animal, dataset_type, p_LD_influence) = entry
        # result_dict = dict()

        eval_path = fix_variables[work_from][animal]['eval_positions']
        print(eval_path, flush=True)
        linkage_data = pkl.load(open(eval_path, 'rb'))
        if len(linkage_data['result']) == 12:  # isinstance(linkage_data,dict):
            idx_list_pkl_data = np.asarray(linkage_data['selected']).tolist() + np.asarray(
                linkage_data['unselected']).tolist()
            idx_list = list(range(len(idx_list_pkl_data)))
            [eval_positions, afc, linkage_def1, linkage_def_kramer, no_window_linkage_def, tmp_distance,
             afc_neighbor_metric, afc_neighbor_linkage_corr, all_n_f, r2, d_normal, r_no_sq] = linkage_data['result']
            linkage_trust_dict = np.asarray([])
        else:
            [eval_positions_, linkage_trust_dict, sf_filter, all_n_f, all_n_f_rsquare, all_n_f_D_normal,
             all_n_f_r_no_square] = linkage_data['result']
            idx_list_pkl_data = np.asarray(eval_positions_['selected'][chrom]).tolist() + np.asarray(
                eval_positions_['unselected'][chrom]).tolist()
            idx_list = list(range(len(idx_list_pkl_data)))
            r2 = all_n_f_rsquare[chrom]
            d_normal = all_n_f_D_normal[chrom]
            all_n_f = all_n_f[chrom]
            r_no_sq = all_n_f_r_no_square[chrom]
            # linkage_trust_dict=np.asarray(linkage_trust_dict[chrom])
            linkage_trust_dict = linkage_trust_dict[chrom]
            # print('linkege trust dict shape',linkage_trust_dict.shape)

        pkl_data = pkl.load(open(fix_variables[work_from][animal_noise]['sync_process'], 'rb'))
        gt_freqs = pkl_data['alleles']
        distances = pkl_data['distances']
        nucleotides = pkl_data['nucleotide_idx']
        if isinstance(gt_freqs, dict):
            gt_freqs = np.squeeze(gt_freqs[chrom])
            distances = np.squeeze(distances[chrom])
            nucleotides = np.squeeze(nucleotides[chrom])

        gt_freqs = np.transpose(gt_freqs, (2, 0, 1))  # [1:]

        print('gt_freqs.shape,len(idx_list_pkl_data)', gt_freqs.shape, len(idx_list_pkl_data))

        # np.set_printoptions(threshold=sys.maxsize)
        # print('gt_freqs sample first 500 pos',np.transpose(gt_freqs[0:afc_gen+1,:500],(1,2,0)))

        naive_sim_scores, valid_indices, val_indices2 = calculate_naive_sim_scores(gt_freqs, idx_list_pkl_data,
                                                                                   # gt_freqs
                                                                                   use_abs=use_abs,
                                                                                   num_train_gens=num_train_gens)
        linkage_trust_arr = []

        if time_series_LD:
            # TODO
            ER_haplo_base_path = fix_variables[work_from][animal]['ER_haplos']
            out_file = ER_haplo_base_path + '/LD_ER_merge.pkl'
            dna_target_positions, dna_neigbor_positions, LD_result_merge = pkl.load(open(out_file, 'rb'))
            # (target_positions,4,100,generations,replicates)
            linkage_defs = np.mean(np.mean(LD_result_merge[:, linkage_idx, :, 0:afc_gen + 1], axis=2), axis=2)
        else:
            linkage_defs = [all_n_f, r2, d_normal, r_no_sq][linkage_idx]

        freqs_focus_once = gt_freqs[0, :, 0]  # shape (replicates,num_snps)
        afc = np.abs(gt_freqs[0] - gt_freqs[afc_gen])

        freqs_focus = []  # shape (replicates,num_snpsx100)
        idx_neighbors = []  # shape (replicates,num_snpsx100)
        idx_focus = []
        afc_neighbor = []
        actual_linkage = []
        # TODO filter unvalid indices DONE
        val_indices2 = []
        val_indices1 = []
        r2_filter = []
        for i, idx in enumerate(idx_list_pkl_data):
            if idx < 50 or idx + 51 > gt_freqs.shape[1]:
                print('skip', idx, i)
                continue
            neighbor_idx = np.concatenate([np.arange(idx - 50, idx), np.arange(idx + 1, idx + 51)])
            idx_neighbors.extend(neighbor_idx)
            idx_focus.extend([idx] * 100)
            neighbor_afc_rep = afc[neighbor_idx, :]
            afc_neighbor.append(neighbor_afc_rep)

            val_indices2.append(idx)
            val_indices1.append(i)

            entry = linkage_defs[i]
            actual_linkage.extend(entry.tolist())
            r2_filter.extend(r_no_sq[i].tolist())
            # columns_link = ['dataset_name', 'dataset_type', 'p_{LD} influence', 'Linkage']
            for e in entry:
                linkage_rows.append({
                    'dataset_name': ds_name,
                    'dataset_type': dataset_type,
                    'p_{LD} influence': p_LD_influence,
                    'Linkage': e,
                    'Chrom': chrom
                })
            if len(linkage_trust_dict):
                linkage_trust_arr.extend(linkage_trust_dict[i])
                if not len(linkage_trust_dict[i]):
                    print(i, idx, 'ERROR nothing inside ld trust dict')
        idx_list = val_indices1
        afc_neighbor = np.concatenate(afc_neighbor, axis=0)


        afc_focus = np.repeat(afc[np.asarray(val_indices2), :], repeats=100, axis=0)

        idx_focus_arr = np.asarray(idx_focus, dtype=int)
        idx_neighbors_arr = np.asarray(idx_neighbors, dtype=int)
        sf_focus_pairs = np.repeat(gt_freqs[0, np.asarray(val_indices2), :], repeats=100, axis=0)
        sf_neighbor_pairs = gt_freqs[0, idx_neighbors_arr, :]


        signed_change = (gt_freqs[afc_gen] - gt_freqs[0])

        delta_focus_pairs = np.repeat(signed_change[np.asarray(val_indices2), :], repeats=100, axis=0)

        delta_neighbor_pairs = signed_change[idx_neighbors_arr, :]

        # physical distance for every exact SNP pair
        pair_distance = np.abs(
            np.asarray(distances[idx_focus_arr], dtype=float) - np.asarray(distances[idx_neighbors_arr], dtype=float))

        print('VAE+LDx pair features:', sf_focus_pairs.shape, sf_neighbor_pairs.shape, delta_focus_pairs.shape,
              delta_neighbor_pairs.shape, pair_distance.shape)

        linkage_trust_arr = np.asarray(linkage_trust_arr)
        actual_linkage = np.asarray(actual_linkage)
        print('actual_linkage', actual_linkage.shape, afc_focus.shape, linkage_trust_arr.shape)
        #################################################################
        # r aufteilung nach vorzeichen + debugging
        r2_arr_obj = np.asarray(r2_filter, dtype=object)

        none_mask = np.array([x is None for x in r2_arr_obj])
        n_none = np.sum(none_mask)

        print("Total number of r2 values:", len(r2_arr_obj))
        print("Number of None values:", n_none)
        print("Fraction of None values:", n_none / len(r2_arr_obj))

        # Positionen der None-Werte
        none_indices = np.where(none_mask)[0]

        print("First None indices:")
        print(none_indices[:20])

        r2_arr = np.array(
            [np.nan if x is None else float(x) for x in r2_arr_obj],
            dtype=float
        )

        valid_mask = ~np.isnan(r2_arr)

        r2_mask_pos = valid_mask & (r2_arr >= 0)
        r2_mask_neg = valid_mask & (r2_arr < 0)

        afc_joint_reps = np.minimum(afc_focus, afc_neighbor)
        if False:
            afc_joint = np.mean(afc_joint_reps, axis=1)
        else:
            afc_joint = np.minimum(np.mean(afc_focus, axis=1), np.mean(afc_neighbor, axis=1))

        LDx_gt_r_square = None
        LDx_result_mask = None
        LDx_joint_depth = None
        LDx_replicates = None
        LDx_masks = []
        if linkage_idx == 1:
            if 'LDx' in fix_variables[work_from][animal_noise]:
                LDx_paths = fix_variables[work_from][animal_noise]['LDx'][chrom]
                LDx_str_ps = read_LDx(LDx_paths, '.txt', ld_col=12)
                # LDx_str_ps = read_LDx_r(LDx_paths,nucleotide_idxs=nucleotides,idx_focus=idx_focus,distances=distances,idx_neighbors=idx_neighbors ,filter_name='.txt')
                LDx_gt_r_square, LDx_result_mask, LDx_joint_depth = get_LDx_intersections(val_indices1, val_indices2,
                                                                                          LDx_str_ps, distances)
        elif linkage_idx == 3:
            if 'LDx' in fix_variables[work_from][animal_noise]:
                LDx_paths = fix_variables[work_from][animal_noise]['LDx'][chrom]
                # LDx_str_ps = read_LDx(LDx_paths, '.txt', ld_col=12)
                LDx_str_ps = read_LDx_r(LDx_paths, nucleotide_idxs=nucleotides, idx_focus=idx_focus,
                                        distances=distances, idx_neighbors=idx_neighbors, filter_name='.txt')
                LDx_gt_r_square, LDx_result_mask, LDx_joint_depth = get_LDx_intersections(val_indices1,
                                                                                          val_indices2,
                                                                                          LDx_str_ps, distances)
        if linkage_idx == 1 or linkage_idx == 3:

            # Only needed for replicatewise training.
            # Pairwise mode does NOT read the LDx files a second time.
            if (run_VAE_LDx and vae_ldx_learning_mode == 'replicatewise'):
                LDx_replicates, LDx_depth_replicates = read_LDx_r_replicates(LDx_paths, nucleotide_idxs=nucleotides,
                                                                             idx_focus=idx_focus,
                                                                             idx_neighbors=idx_neighbors,
                                                                             distances=distances,
                                                                             val_indices1=val_indices1,
                                                                             val_indices2=val_indices2,
                                                                             filter_name='.txt')

                print('LDx replicate matrix:', LDx_replicates.shape)

            if LDx_gt_r_square.shape != actual_linkage.shape:
                raise ValueError('LDx and ground-truth pair arrays are not aligned')

            LDx_base_mask = LDx_result_mask.copy()
            if len(linkage_trust_dict):
                LDx_base_mask &= (linkage_trust_arr == 1.0)

            # Set thresholds once, using pairs eligible for evaluation.
            # Ground-truth LD magnitudes and correlations do not set them.
            LDx_eval_mask = LDx_base_mask
            LDx_num_unfiltered = np.count_nonzero(LDx_eval_mask)
            LDx_known_depth = np.isfinite(LDx_joint_depth) & (LDx_joint_depth >= 0)
            LDx_quantiles = np.linspace(0.0, 0.90, 20)
            LDx_depth_thresholds = np.full(20, np.inf)
            LDx_quantile_mask = LDx_eval_mask & LDx_known_depth
            if np.any(LDx_quantile_mask):
                LDx_depth_thresholds = np.quantile(
                    LDx_joint_depth[LDx_quantile_mask], LDx_quantiles)
            LDx_depth_thresholds[0] = 0.0  # Label only: level 0 is unfiltered.
            if np.any(LDx_eval_mask & ~LDx_known_depth):
                print('LDx pairs with unknown depth (kept at level 0):',
                      np.count_nonzero(LDx_eval_mask & ~LDx_known_depth))

            for afc_diff_idx in range(20):
                depth_threshold = LDx_depth_thresholds[afc_diff_idx]
                if afc_diff_idx == 0:
                    mask = LDx_base_mask.copy()
                else:
                    mask = (LDx_base_mask & LDx_known_depth
                            & (LDx_joint_depth > 0)
                            & (LDx_joint_depth >= depth_threshold))
                mask_pos = mask & r2_mask_pos
                mask_neg = mask & r2_mask_neg

                num_true_pos = np.count_nonzero(mask_pos)
                num_true_neg = np.count_nonzero(mask_neg)
                num_true = num_true_pos + num_true_neg

                if num_true_pos < 30 or num_true_neg < 30:
                    print('LDx: stopping at filter', afc_diff_idx,
                          '(fewer than 30 pairs in a sign group)',
                          num_true_pos, num_true_neg)
                    break
                print('LDx filter/depth/positive/negative:', afc_diff_idx,
                      depth_threshold, num_true_pos, num_true_neg)
                LDx_masks.append((mask, mask_pos, mask_neg, num_true, num_true))
                sp, pears_ldx, mae_ldx, rmse_ldx, xi = calculate_corr_values(actual_linkage[mask_pos],
                                                                             LDx_gt_r_square[
                                                                                 mask_pos])  # np.asarray(r2).flatten()[mask]
                sp_neg, pears_ldx_neg, mae_ldx_neg, rmse_ldx_neg, xi_neg = calculate_corr_values(
                    actual_linkage[mask_neg],
                    LDx_gt_r_square[mask_neg])

                sp = summarize_values(sp, mask_pos, sp_neg, mask_neg)
                pears_ldx = summarize_values(pears_ldx, mask_pos, pears_ldx_neg, mask_neg)
                mae_ldx = summarize_values(mae_ldx, mask_pos, mae_ldx_neg, mask_neg)
                rmse_ldx = summarize_values(rmse_ldx, mask_pos, rmse_ldx_neg, mask_neg)

                same_sign_percentage = np.mean(
                    np.sign(actual_linkage[mask]) == np.sign(LDx_gt_r_square[mask])) * 100
                print('same_sign_percentage', same_sign_percentage)
                sp_norm, pears_ldx_norm, mae_ldx_norm, rmse_ldx_norm, xi_norm = calculate_corr_values(
                    actual_linkage[mask],
                    LDx_gt_r_square[mask])

                afc_rows.append({
                    'dataset_name': ds_name,
                    'dataset_type': dataset_type,
                    'p_{LD} influence': p_LD_influence,
                    'AFC filter': afc_diff_idx * 0.01,
                    'LD estimator': 'LDx',
                    'Chrom': chrom,
                    'Spearman roh diffcomp': sp,
                    'Pearson diffcomp': pears_ldx,
                    'MAE diffcomp': mae_ldx,
                    'RMSE diffcomp': rmse_ldx,
                    'Spearman roh': sp_norm,
                    'Pearson': pears_ldx_norm,
                    'MAE': mae_ldx_norm,
                    'RMSE': rmse_ldx_norm,
                    'XI': xi,
                    'num_eval_spns': num_true,
                    'num_all_spns': np.count_nonzero((linkage_trust_arr == 1.0)) if len(linkage_trust_dict) else
                    actual_linkage.shape[0]

                })

                print('LDx num true', num_true, sp)

        print(afc_neighbor.shape, afc_focus.shape)
        # direction_similarity = np.abs(afc_neighbor - afc_focus)


        use_afc_weights = False  # False: bisheriger Mean
        afc_weight_sum = afc_joint_reps.sum(axis=1)

        for submodel_num in range(num_submodels):
            model_name_ = model_name + '_' + str(submodel_num) + '_fine'
            path = tmp_data_folder + 'models/' + model_name_ + '/eval_data/train_data.pkl'
            in_stream = open(path, 'rb')
            data = pkl.load(in_stream)

            potential_linkage_data = data['window_representation']
            if isinstance(potential_linkage_data, dict):
                potential_linkage_data = potential_linkage_data[
                    chrom]  # shape p.zeros((1,len(idx_list), sample_size, 1+(dna_vae.snp_window_half*2)))
            print('model values', potential_linkage_data.shape, len(idx_list), idx_list[:5], chrom, flush=True)
            in_stream.close()
            potential_linkage_data = np.squeeze(potential_linkage_data)[idx_list]
            potential_linkage_data = potential_linkage_data.transpose(0, 2, 1)

            neighbor_diff = gt_ld_neighbors - num_neighbors
            if potential_linkage_data.shape[1] % 2 == 1:
                pot_linkage = []
                if neighbor_diff == 0:
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry[:50].tolist())
                        pot_linkage.extend(entry[51:].tolist())
                elif neighbor_diff < 0:
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry[-neighbor_diff:-neighbor_diff + gt_ld_neighbors].tolist())
                        pot_linkage.extend(entry[-neighbor_diff + gt_ld_neighbors + 1:-neighbor_diff + (
                                2 * gt_ld_neighbors) + 1].tolist())
                else:
                    # num_neighbors is smaler
                    print('not implemented - as case does not exist so far')
                    pass

                pot_linkage = np.asarray(pot_linkage)
            else:
                pot_linkage = []
                if neighbor_diff == 0:
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry.tolist())
                elif neighbor_diff < 0:
                    # num_neighbors is larger
                    for entry in potential_linkage_data:
                        pot_linkage.extend(entry.tolist()[-neighbor_diff:-neighbor_diff + (2 * gt_ld_neighbors)])
                else:
                    # num_neighbors is smaler
                    print('not implemented - as case does not exist so far')
                    pass

                pot_linkage = np.asarray(pot_linkage)


            model_rows = calculate_VAE_LDx_regression(
                pot_linkage=pot_linkage,
                LDx_values=LDx_gt_r_square,
                LDx_result_mask=LDx_result_mask,
                actual_linkage=np.asarray(actual_linkage, dtype=float),
                gt_freqs=gt_freqs,
                idx_focus=idx_focus,
                idx_neighbors=idx_neighbors,
                distances=distances,
                afc_gen=afc_gen,
                linkage_trust_arr=linkage_trust_arr,
                linkage_trust_dict=linkage_trust_dict,
                r2_mask_pos=r2_mask_pos,
                r2_mask_neg=r2_mask_neg,
                model_rows=model_rows,
                ds_name=ds_name,
                dataset_type=dataset_type,
                p_LD_influence=p_LD_influence,
                submodel_num=submodel_num,
                model_name=model_name,
                plot_name=plot_name,
                chrom=chrom,
                filter_modes=['coherence', 'afc_min', 'afc_geom', 'afc_max', 'afc_sf_min'],
                # , 'coherence', 'afc_min' 'none'
                train_fraction=0.80,
                max_train_pairs=50000,
                random_state=42,
                linkage_idx=linkage_idx
            )
            model_rows, afc_rows = calculate_LDcorrelation(pot_linkage, linkage_trust_arr, linkage_trust_dict,
                                                           afc_joint_reps, actual_linkage, afc_focus, afc_neighbor,
                                                           afc_weight_sum, data,
                                                           chrom, val_indices2, idx_neighbors, naive_sim_scores,
                                                           model_rows, ds_name, dataset_type, p_LD_influence,
                                                           submodel_num, model_name, afc_rows, plot_path, linkage_idx,
                                                           plot_name, r2_mask_pos, r2_mask_neg,
                                                           idx_focus=idx_focus, gt_freqs=gt_freqs, afc_gen=afc_gen,
                                                           LDx_masks=LDx_masks,
                                                           filter_modes=['coherence', 'afc_min', 'afc_geom', 'afc_max',
                                                                         'afc_sf_min', 'LDx_mask'])

        print('plot data')

    return model_rows, afc_rows, linkage_rows








