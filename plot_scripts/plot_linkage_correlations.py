import os
import pickle as pkl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import matplotlib


matplotlib.use('TkAgg')

sns.set_style("darkgrid", {'axes.grid': True})
sns.set(font_scale=1.0,
        rc={"text.usetex": True,
        "font.family":'sans-serif',#"Times",# "Arial",#"serif",
        "font.sans-serif": "Helvetica",
        'font.size':12,#10,
        'axes.titlesize':12,
        'axes.labelsize':12,#10
            })
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] ='sans-serif'# "Times",#"Arial",#'serif'
plt.rcParams['font.sans-serif'] ="Helvetica"
plt.rcParams['font.size'] = 12#10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12#10




sns.set_style("whitegrid")

def plot_correlations(df_model_, df_afc_, columns_model, columns_afc,linkage_name,spearman_type_='max',use_old=False):
    # Define the grid layout
    subplot_labels = ['A', 'B']
    dataset_type=['no noise','(100,40)']#'(200,50)',

    #######################################################
    # calculate model improvement from bestcase to worstcase:

    df_model = df_model_.copy()
    df_filter = df_model[df_model['dataset_type'] == '(100,40)']


    df_model_best_case = df_filter.groupby(
        ['dataset_name', 'dataset_type', 'p_{LD} influence', 'submodel_num', 'LD estimator', 'model name']
        # ,'model plot name'
    ).agg(max_spearman=('Spearman roh', 'max')).reset_index()

    df_model_worst_case = df_filter[df_filter['AFC filter'] == 0].copy()
    df_model_worst_case.rename(columns={'Spearman roh': 'max_spearman'}, inplace=True)

    num_targets=df_model_worst_case['p_{LD} influence'].unique()
    LD_noises =df_model_worst_case['dataset_name'].unique()
    submodel_nums=df_model_worst_case['submodel_num'].unique()

    improvements=[]
    abs_improvement=[]
    for target in num_targets:
        for ld_noise in LD_noises:
            avg_best=0
            avg_worst=0
            for submodel_num in submodel_nums:
                spear_worst_case = df_model_worst_case[
                    (df_model_worst_case['p_{LD} influence'] == target) &
                    (df_model_worst_case['dataset_name'] == ld_noise) &
                    (df_model_worst_case['submodel_num'] == submodel_num)
                    ]['max_spearman']
                spear_best_case = df_model_best_case[
                    (df_model_best_case['p_{LD} influence'] == target) &
                    (df_model_best_case['dataset_name'] == ld_noise) &
                    (df_model_best_case['submodel_num'] == submodel_num)
                    ]['max_spearman']
                avg_best+=spear_best_case.values[0]
                avg_worst+=spear_worst_case.values[0]
            avg_worst/=len(submodel_nums)
            avg_best/=len(submodel_nums)
            improvements.append(avg_best/avg_worst)
            abs_improvement.append(avg_best-avg_worst)
    print(improvements)
    improvements=np.asarray(improvements)
    print(np.mean(improvements),np.min(improvements),np.max(improvements))
    abs_improvement=np.asarray(abs_improvement)
    print(np.mean(abs_improvement), np.min(abs_improvement), np.max(abs_improvement))






    #####################################################

    #fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    for j, (ds_type, path_type) in enumerate(zip(dataset_type, ['gt', '100_40'])):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for i,spearman_type in enumerate(['Worst case','Best case']):
            ax=axes[i]
            #fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharey=True)
            df_afc=df_afc_.copy()
            df_model=df_model_.copy()


            source = 'LD estimator'#'LD estimator'
            df_afc=df_afc[df_afc[source]!='afc']

            df_afc['Spearman roh'] = df_afc['Spearman roh'].abs()
            df_model['Spearman roh'] = df_model['Spearman roh'].abs()


            if spearman_type=='Best case':
                df_model_grouped = df_model.groupby(
                    ['dataset_name', 'dataset_type', 'p_{LD} influence', 'submodel_num','LD estimator','model name']#,'model plot name'
                ).agg(max_spearman=('Spearman roh', 'max')).reset_index()

                df_afc_grouped = df_afc.groupby(['dataset_name', 'dataset_type', 'p_{LD} influence','LD estimator']).agg(
                    max_spearman=('Spearman roh', 'max')
                ).reset_index()
            else:
                # no_filter
                #'AFC filter'==0
                #max_spearman
                df_model_grouped=df_model[df_model['AFC filter'] == 0].copy()
                df_model_grouped.rename(columns={'Spearman roh': 'max_spearman'}, inplace=True)

                df_afc_grouped=df_afc[df_afc['AFC filter'] == 0].copy()
                df_afc_grouped.rename(columns={'Spearman roh': 'max_spearman'}, inplace=True)



            df_model_grouped[source]='VAE' #df_model_grouped['model plot name']#
            #df_model_grouped[source]=df_model_grouped['model plot name']
            df_afc_grouped['submodel_num'] = 0
            df_afc_grouped['model name']=''

            # 3. Combine Data
            df_combined = pd.concat([df_model_grouped, df_afc_grouped])

            df_combined['dataset_type'] = df_combined['dataset_type'].replace('(1000,1000)', 'no noise')
            def convert_labels0(label):
                label = re.sub(r'N\((.*?)\)', r'\\mathcal{N}(\1)', label)  # Convert 'N(xy)' to '\mathcal{N}(xy)'

                return f'${label}$'  # Add outer LaTeX dollar signs
            def convert_labels1(label):
                label = re.sub(r'(p_{LD}=.*?)', r'\1', label)  # Convert 'p_{LD}=xy' to '\p_{LD}=xy'
                return f'${label}$'  # Add outer LaTeX dollar signs

            if use_old:
                df_combined['dataset_name']=df_combined['dataset_name'].apply(convert_labels0)
                df_combined['p_{LD} influence']=df_combined['p_{LD} influence'].apply(convert_labels1)
                df_combined['x_label'] = df_combined['dataset_name'] + '\n' + df_combined['p_{LD} influence'].astype(str)
                df_combined = df_combined.sort_values(by=[ 'p_{LD} influence','dataset_name' ], ascending=[True, False])

            else:
                df_combined['x_label'] ="$n_{LD}="+df_combined['dataset_name']+"$\n$|T|="+df_combined['p_{LD} influence'].astype(str)+"$"
                #df_combined['x_label'] =df_combined['dataset_name'] + ':' + df_combined['p_{LD} influence'].astype(str)

                df_combined['dataset_name'] = df_combined['dataset_name'].astype(int)
                df_combined = df_combined.sort_values(by=['dataset_name', 'p_{LD} influence', ], ascending=[True, False])



            # Function to convert strings
            # Function to convert strings
            '''
            def convert_labels(label):
                label = re.sub(r'N\((.*?)\)', r'$\mathcal{N}(\g<1>)$', label)  # Convert 'N(xy)' to '$\mathcal{N}(xy)$'
                label = re.sub(r'(p_{LD}=.*?)', r'$\1$', label)  # Convert 'p_{LD}=xy' to '$p_{LD}=xy$'
                return label
            '''

            # Function to convert strings
            def convert_labels(label):
                label = re.sub(r'N\((.*?)\)', r'$\\mathcal{N}(\1)$', label)  # Convert 'N(xy)' to '\mathcal{N}(xy)'
                label = re.sub(r'(p_{LD}=.*?)', r'$\\\1$', label)  # Convert 'p_{LD}=xy' to '\p_{LD}=xy'
                return f'${label}$'  # Add outer LaTeX dollar signs

            # Apply the function to the column
            #df_combined['x_label'] = df_combined['x_label'].apply(convert_labels)

            # 5. Sort x-axis labels alphabetically



            # 6. Plot using Seaborn's built-in error bars
            #plt.figure(figsize=(12, 6))
            # Create subplots

            df_combined[source] = df_combined[source].replace({'org_sim': 'Baseline'})
            #'dataset_type'
            # Define custom colors for each category
            default_palette = sns.color_palette("tab10")
            custom_palette = {'VAE': default_palette[1],  # Orange
                          'LDx': default_palette[2],  # Green
                          'Baseline': default_palette[0]}  # Blue

            df_filter=df_combined[df_combined['dataset_type']==ds_type]
            sns.pointplot(
                data=df_filter,
                x='x_label',
                y='max_spearman',
                hue=source,
                #style='dataset_type',
                #markers=['o', 's'],  # Different markers for different sources
                ax=ax,
                dodge=True,  # Avoid overlapping points
                #errorbar="sd",  # Seaborn automatically computes standard deviation for error bars
                #capsize=0.1
                linestyles='',
                palette=custom_palette  # Apply custom colors
            )
            ax.grid(True, which='both', linestyle='--', linewidth=0.7)

            # Labels and legend
            #axes[i].set_xlabel("Datasets with descending LD (defined with D')")
            #axes[i].set_xlabel("Datasets with LD noise %:#targets")
            ax.set_xlabel("Dataset configuration")

            ax.set_ylabel("Spearman's $\\rho$")
            ax.set_title(spearman_type)

            #print(axes[i].get_xticks(),axes[i].xaxis.get_majorticklabels(),axes[i].xaxis.get_minorticklabels())
            #axes[i].set_xticks(axes[i].xaxis.get_majorticklabels(),rotation=45, ha='right')
            #plt.xticks(rotation=45, ha='right')
        #plt.legend(title=linkage_name)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True)
            fig.text(0.06 + i * 0.49, 0.89 , subplot_labels[i ], fontsize=12, weight='bold')#- i * 0.418
            if i == 0:
                ax.legend().remove()
            else:
                ax.legend(title='LD estimator',loc='upper right')#title="Method",

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave some space at the top for subplot labels
        #plt.grid(True)

            #plt.tight_layout()
        fig.savefig(plot_path + path_type+ '_link_complete.png')
        fig.savefig(plot_path + path_type + '_link_complete.pdf')
        fig.savefig(plot_path + path_type+ '_link_complete.tiff')
            #plt.show()


def plot_performance(plot_path,use_old=False):
    subplot_labels=['A','B','C','D']
    data_summary_path = plot_path +'simulation_results.pkl'
    df_org = pkl.load(open(data_summary_path, 'rb'))
    #print(df.head(20))

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    #filter one DS
    df_org = df_org[~((df_org['dataset_name'] == 'N(.6,.5)') &
                      (df_org['p_{LD} influence'] == 'p_{LD}=.0') &
                      (df_org['dataset_type'] == 'no noise'))]
    df_org['dataset_type'] = df_org['dataset_type'].replace('(1000,1000)', 'no noise')

    for data_type,path_type in zip(['no noise','(200,50)','(100,40)'],['gt','200_50','100_40']):
        df=df_org[df_org['dataset_type']==data_type]

        # Define the grid layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey='row')

        # Define conditions for subplots
        result_metrics = ['relative_mean_error', 'relative_std_error']#'relative_mean_error', 'relative_std_error']
        snp_types = ['selected', 'unselected']
        snp_types_print=['Targets','No targets']
        result_metrics_plot=['mean','standard deviation']
        def convert_labels0(label):
            label = re.sub(r'N\((.*?)\)', r'\\mathcal{N}(\1)', label)  # Convert 'N(xy)' to '\mathcal{N}(xy)'

            return f'${label}$'  # Add outer LaTeX dollar signs

        def convert_labels1(label):
            label = re.sub(r'(p_{LD}=.*?)', r'\1', label)  # Convert 'p_{LD}=xy' to '\p_{LD}=xy'
            return f'${label}$'  # Add outer LaTeX dollar signs
        if use_old:
            df['dataset_name']=df['dataset_name'].apply(convert_labels0)
            df['p_{LD} influence']=df['p_{LD} influence'].apply(convert_labels1)
            df['x_label'] = df['dataset_name'] + '\n' + df['p_{LD} influence']
            #df['dataset_name'] = df['dataset_name'].astype(int)
            df= df.sort_values(by=['p_{LD} influence', 'dataset_name'], ascending=[True, False])#[True, False]
        else:
            # Prepare the dataset
            df['x_label'] = "$n_{LD}=" + df['dataset_name'] + "$\n$|T|=" + df[
                'p_{LD} influence'].astype(str) + "$"
            #df['x_label'] = df['dataset_name'] + ':' + df['p_{LD} influence'].astype(str)
            df['dataset_name'] = df['dataset_name'].astype(int)
            df= df.sort_values(by=['dataset_name','p_{LD} influence'], ascending=[True, False])#[True, False]

        default_palette = sns.color_palette("tab10")
        #custom_palette = {'VAE': default_palette[0],  # Orange
        #                  'LDx': default_palette[2],  # Green
        #                  'Baseline': default_palette[1]}  # Blue


        # Define colors for models
        model_colors = {
            'WF model': 'red',
            'w': default_palette[1],
            'no w': default_palette[0]  # Replace with actual model names
        }
        snp_type_reverse=[snp_types[1],snp_types[0]]
        # Create subplots
        for i, result_metric in enumerate(result_metrics):
            for j, snp_type in enumerate(snp_types):
                ax = axes[i, j]


                df_filtered = df[(df['result_metric_name'] == result_metric) & (df['snp type'] == snp_type)]
                print(df['result_metric'] )

                sns.pointplot(
                    data=df_filtered,
                    x='x_label',
                    y='result_metric',
                    hue='model name',
                    ax=ax,
                    markers=['x' if model == 'WF model' else 'o' for model in df_filtered['model name'].unique()],
                    palette=[model_colors.get(model, 'gray') for model in df_filtered['model name'].unique()],
                    dodge=True,
                    linestyles=''
                )
                # Force grid lines on both x and y axes
                ax.grid(True, which='both', linestyle='--', linewidth=0.7)

                # Titles
                #ax.set_title(f"{result_metric.replace('_', ' ').title()} - SNP Type: {snp_types[j]}")#{snp_type_reverse[j]}
                if i==0:
                    ax.set_title(snp_types_print[j])  # {snp_type_reverse[j]}
                else:
                    ax.set_title('')
                # Customize legend
                if i==0 or j==0:
                    ax.legend().remove()
                else:
                    ax.legend(title="Method",loc='upper right')

                # Rotate x-axis labels for better visibility
                print(ax.get_xticklabels())
                #if use_old:
                #    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",fontsize=8)

                ax.set_xlabel("Dataset configuration")#("Datasets "+data_type+" with LD noise %:#targets")
                #ax.set_ylabel(result_metric.replace('_', ' ').title())
                ax.set_ylabel('Relative distance: '+result_metrics_plot[i])
                #ax.tick_params(axis='both', labelsize=7)  # Tick labels size
                #ax.set_xticklabels(rotation=45, ha='right')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                #print(ax.get_xticklabels())
                fig.text(0.06 + j * 0.48, 0.96 - i * 0.418, subplot_labels[i*2+j], fontsize=12, weight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave some space at the top for subplot labels
        # Adjust layout
        plt.tight_layout()

        fig.savefig(plot_path +path_type+ '_gen_perf_complete.png')
        fig.savefig(plot_path +path_type+ '_gen_perf_complete.pdf')
        fig.savefig(plot_path +path_type+ '_gen_perf_complete.tiff')
        plt.show()







if __name__ == '__main__':
    columns_model = ['dataset_name', 'dataset_type','p_{LD} influence','model name', 'submodel_num','AFC filter','Spearman roh']
    columns_afc = ['dataset_name', 'dataset_type','p_{LD} influence','AFC filter','Spearman roh']
    plot_path='plots/compare_all/'
    use_old=False
    plot_performance(plot_path,use_old=use_old)
    
    for l_idx , linkage_name in enumerate(["$D'$", '$r^2$', '$D$']):
        if l_idx in [1]:
            data_summary_path = plot_path +str(l_idx) +'LD_spearman_corr.pkl'
            print(data_summary_path)
            # Check if the file exists and its size
            if not os.path.exists(data_summary_path):
                print("Error: File does not exist!")
            elif os.path.getsize(data_summary_path) == 0:
                print("Error: File is empty!")
            else:
                print("File exists and is not empty.")
            [df_model, df_afc,df_link] = pkl.load(open(data_summary_path, 'rb'))

            plot_correlations(df_model, df_afc,columns_model,columns_afc,linkage_name,spearman_type_='no_filter',use_old=use_old)


        