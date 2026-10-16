import re
import pickle as pkl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
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




'''

def plot_synthetic_max_corr(plot_paths,correlation_key,correlation_plot_key,min_max_best,model_name_filter=[],
                            rename=dict(),plot_path='',noise_str='',filter_mode='',show_mode='max',
                            add_quality_coverage_panels=3,rho_target=0.75,filter_model_names=[]):
    eval_count_col = 'num_eval_spns'
    all_count_col = 'num_all_spns'
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42
    })

    # Define the grid layout
    n_panels = add_quality_coverage_panels
    subplot_labels = ['A', 'B', 'C', 'D'][:n_panels]
    fig, axes = plt.subplots(
        1,
        n_panels,
        #figsize=(3.4*add_quality_coverage_panels, 4),
        figsize=(2.1 * add_quality_coverage_panels, 3),
        sharey=False
    )

    #fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=False)
    #fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    default_palette = sns.color_palette("tab10")

    custom_palette_ = {
        'Baseline': "#4C72B0",  # Blue
        'VAE': "#DD8452",  # Orange
        'LDx':"#C44E52"  ,  # Red
        'random':"#55A868",  # Green
        'VAE+LDx':"#55A868"
    }

    source = 'LD estimator'


    df_summaries=[]
    coverage_summaries = []
    model_names=set()
    # load data

    LD_idx_set=[1,3]
    if show_mode=='r^2':
        LD_idx_set=[1]
    elif show_mode=='r':
        LD_idx_set=[3]
    #
    for j, l_idx in enumerate(LD_idx_set):
        df_summaries.append([])
        coverage_frames = []

        for i, spearman_type in enumerate(['Worst case', 'Best case']):#['Performance on all SNPs', 'Maximum performance after SNP filtering']
            all_frames = []
            for k, (plot_path, vae_add) in enumerate(plot_paths):
                data_summary_path = plot_path + str(l_idx) + 'LD_spearman_corr.pkl'
                #print(os.path.getsize(data_summary_path) / 1024 ** 2, "MB")
                [df_model_, df_afc_, df_link_] = pkl.load(open(data_summary_path, 'rb'))

                df_afc = df_afc_.copy()
                df_model = df_model_.copy()



                if len(model_name_filter):
                    keep_models = set(model_name_filter)
                    df_model = df_model.loc[df_model['model plot name'].isin(keep_models)].copy()


                if 'filter' in df_model.columns:
                    df_model = df_model.loc[ (df_model['filter'] == filter_mode)].copy()#(df_model['LD estimator'] != 'VAE+LDx') |
                    if 'filter' in df_afc.columns:
                        df_afc = df_afc.loc[(df_afc[source] != 'org_sim') | (df_afc['filter'] == filter_mode)].copy()

                #df_model[source] = df_model['model plot name']
                mask = df_model[source] != 'VAE+LDx'
                df_model.loc[mask, source] = df_model.loc[mask, 'model plot name']

                df_afc = df_afc.loc[df_afc[source] != 'afc'].copy()

                df_afc['Spearman roh'] = df_afc['Spearman roh'].abs()
                df_model['Spearman roh'] = df_model['Spearman roh'].abs()
                if spearman_type == 'Best case':
                    df_model_grouped = df_model.groupby(
                        ['dataset_name', 'dataset_type', 'p_{LD} influence', 'submodel_num', 'LD estimator',
                         'model name']  # ,'model plot name'
                    ).agg(max_spearman=(correlation_key, min_max_best)).reset_index()

                    df_afc_grouped = df_afc.groupby(
                        ['dataset_name', 'dataset_type', 'p_{LD} influence', 'LD estimator']).agg(
                        max_spearman=(correlation_key, min_max_best)
                    ).reset_index()
                else:

                    df_model_grouped = df_model[df_model['AFC filter'] == 0].copy()
                    df_model_grouped.rename(columns={correlation_key: 'max_spearman'}, inplace=True)

                    df_afc_grouped = df_afc[df_afc['AFC filter'] == 0].copy()
                    df_afc_grouped.rename(columns={correlation_key: 'max_spearman'}, inplace=True)

                # df_model_grouped[source]='VAE'
                df_afc_grouped['submodel_num'] = 0
                df_afc_grouped['model name'] = ''


                # 3. Combine Data
                df_combined = pd.concat([df_model_grouped, df_afc_grouped])
                df_combined['dataset_type'] = df_combined['dataset_type'].replace('(1000,1000)', 'no noise')

                df_combined[source] = df_combined[source].replace({'org_sim': 'Baseline'})
                df_combined['x_label'] ="$n_{LD}="+df_combined['dataset_name']+"$\n$|T|="+df_combined['p_{LD} influence'].astype(str)+"$"
                #df_combined['x_label'] =df_combined['dataset_name'] + ':' + df_combined['p_{LD} influence'].astype(str)

                df_combined['dataset_name'] = df_combined['dataset_name'].astype(int)
                df_combined = df_combined.sort_values(by=['dataset_name', 'p_{LD} influence', ], ascending=[True, False])

                df_filter = df_combined.loc[df_combined['dataset_type'] == noise_str].copy()

                for key, value in rename.items():
                    if value=='':
                        #delete this:
                        df_filter = df_filter[~df_filter[source].isin([key])].copy()
                    else:
                        # rename this
                        df_filter.loc[:, source] = df_filter[source].replace({key: value})

                if vae_add != '':
                    df_filter.loc[df_filter[source] == "VAE", source] = "VAE" + ' ' + vae_add
                    custom_palette_["VAE " + vae_add] = default_palette[3 + k]
                    # df_filter = df_filter[(df_filter[source] == "VAE" + ' ' + vae_add)]
                    # delete baseline
                    df_filter = df_filter.loc[df_filter[source] != "Baseline"].copy()
                for filter_model_name in filter_model_names:
                    df_filter = df_filter[df_filter[source] != filter_model_name]
                #df_filter = df_filter[df_filter[source] != "LDx"]
                tmp_5 = df_filter[source].unique()

                all_frames.append(df_filter)
                if add_quality_coverage_panels>2 and i == 0:

                    cov_model = df_model.copy()
                    cov_afc = df_afc.copy()

                    # Match structure used later in the existing code
                    cov_afc['submodel_num'] = 0
                    cov_afc['model name'] = ''

                    cov_combined = pd.concat(
                        [cov_model, cov_afc],
                        ignore_index=True,
                        sort=False
                    )

                    cov_combined[source] = (
                        cov_combined[source]
                        .replace({'org_sim': 'Baseline'})
                    )

                    cov_filter = cov_combined.copy()

                    for key, value in rename.items():
                        if value == '':
                            cov_filter = cov_filter.loc[
                                ~cov_filter[source].isin([key])
                            ].copy()
                        else:
                            cov_filter.loc[:, source] = (
                                cov_filter[source]
                                .replace({key: value})
                            )

                    if vae_add != '':
                        cov_filter.loc[
                            cov_filter[source] == "VAE",
                            source
                        ] = "VAE " + vae_add

                        cov_filter = cov_filter.loc[
                            cov_filter[source] != "Baseline"
                        ].copy()

                    # Prevent filter levels from different PKL runs
                    # from being combined before their score is computed.
                    cov_filter['_plot_run'] = k

                    coverage_frames.append(cov_filter)


            global_df = pd.concat(all_frames, ignore_index=True)  #
            tmp=set(global_df[source].unique())
            model_names.update(tmp)
            df_summaries[-1].append(global_df)
        if add_quality_coverage_panels:
            coverage_summaries.append(
                pd.concat(
                    coverage_frames,
                    ignore_index=True
                )
            )

    ld_names=['$r^2$','$r$']
    model_name_vs_ld=dict()

    for model_name in model_names:
        if show_mode == 'r^2':
            model_name_vs_ld[model_name]=0
        elif show_mode == 'r':
            model_name_vs_ld[model_name] = 1
        elif show_mode=='r2_r':
            if model_name=='LDx':
                model_name_vs_ld[model_name]=0
            else:
                model_name_vs_ld[model_name]=1
        else:
            mean_overall_entries_r2 = df_summaries[0][1][df_summaries[0][1][source] == model_name][
                'max_spearman'].mean()
            mean_overall_entries_r1 = df_summaries[1][1][df_summaries[1][1][source] == model_name][
                'max_spearman'].mean()
            if pd.isna(mean_overall_entries_r1):
                mean_overall_entries_r1 = 0
            model_name_vs_ld[model_name]=0 if mean_overall_entries_r2>mean_overall_entries_r1 else 1

    if add_quality_coverage_panels:

        coverage_model_results = []
        coverage_palette = {}

        for model_name, corr_idx in model_name_vs_ld.items():

            # When show_mode='r' or 'r^2', only one dataframe exists.
            if len(LD_idx_set) == 1:
                raw_idx = 0
            else:
                raw_idx = corr_idx

            df_cov = coverage_summaries[raw_idx].loc[coverage_summaries[raw_idx][source] == model_name].copy()

            new_name = model_name + ' vs ' + ld_names[corr_idx]

            df_cov.loc[:, source] = new_name

            coverage_model_results.append(df_cov)

            if model_name in custom_palette_:
                coverage_palette[new_name] = custom_palette_[model_name]

        coverage_global_df = pd.concat(
            coverage_model_results,
            ignore_index=True
        )

        # Colors for any models not explicitly in custom_palette_
        names_in_palette = coverage_palette.keys()

        for n_i, name in enumerate(sorted(set(coverage_global_df[source].unique()).difference(names_in_palette))):
            coverage_palette[name] = default_palette[3 + len(plot_paths) + n_i]

        target_coverage_df, qc_hv_df = _summarize_quality_coverage(
            coverage_global_df,
            source=source,
            correlation_key=correlation_key,
            rho_target=rho_target,
            eval_count_col=eval_count_col,
            all_count_col=all_count_col
        )

    if add_quality_coverage_panels == 3:
        headlines = ['Perf. on all evaluable pairs']  # [r'$\rho$ on all evaluable pairs']
    else:
        headlines = ['Perf. on all evaluable pairs', 'Maximum performance after SNP filtering']

    for i, spearman_type in enumerate(headlines): #['Worst case','Best case']

        ax=axes[i]
        model_results=[]
        custom_palette = {}
        for model_name, corr_idx in model_name_vs_ld.items():
            df=df_summaries[corr_idx][i].loc[df_summaries[corr_idx][i][source]==model_name]
            new_name=model_name+' vs '+ld_names[corr_idx]
            df[source]=new_name
            model_results.append(df)
            if model_name in custom_palette_:
                custom_palette[new_name]=custom_palette_[model_name]
        global_df = pd.concat(model_results, ignore_index=True)

        model_names_in_palette = custom_palette.keys()
        for n_i, name in enumerate(sorted(set(global_df[source].unique()).difference(model_names_in_palette))):
            custom_palette[name] = default_palette[3 + len(plot_paths) + n_i]

        print(custom_palette)

        sns.pointplot(
            data=global_df,
            x='x_label',
            y='max_spearman',
            hue=source,
            ax=ax,
            dodge=0.3,
            markers='o',
            linestyles='',
            #scale=1.0,  # dot size
            #errwidth=1,  # thin error bars
            markersize=6,
            err_kws={'linewidth': 1},
            capsize=0.1,
            palette=custom_palette
        )

        for collection in ax.collections:
            collection.set_edgecolor("white")
            collection.set_linewidth(0.6)

        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.grid(False, which='minor')

        ax.set_xlabel("Dataset configuration")
        if i==0:
            ax.set_ylabel(correlation_plot_key)
        else:
            ax.set_ylabel('')

        ax.set_title(spearman_type)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True)

        #fig.text(0.06 + i * 0.49, 0.87 , subplot_labels[i ], fontsize=10, weight='bold')#- i * 0.418

        ax.legend().remove()









    # Get handles + labels from the last axis
    handles, labels = ax.get_legend_handles_labels()

    # Add global legend at top center
    dummy = Line2D([], [], linestyle='none')

    handles = [dummy] + handles
    labels = ["LD estimator"] + labels
    legend= fig.legend(
        handles,
        labels,
        #title="LD estimator",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        #bbox_to_anchor=(0.5, 1.08)
        ncol=len(labels), #3,  # adjust depending on how many legend entries you have

        handletextpad=0.4,
        columnspacing=1.2,
        frameon=True,
        fancybox = True
    )
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#e0e0e0')
    frame.set_linewidth(1)

    for ax in axes:
        ax.spines['top'].set_linewidth(0.4)#.set_visible(False)
        ax.spines['right'].set_linewidth(0.4)#.set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.tick_params(width=0.8)

    #plt.subplots_adjust(left=0.05, right=0.98, hspace=0.2, bottom=0.2, wspace=0.05)
    #plt.subplots_adjust(top=0.82)
    #plt.subplots_adjust(left=0.08,right=0.98,bottom=0.22,top=0.82,wspace=0.15)
    if add_quality_coverage_panels:
        #plt.subplots_adjust(left=0.06,right=0.99,bottom=0.20,top=0.82,wspace=0.28)
        plt.subplots_adjust(left=0.075, right=0.992, bottom=0.20, top=0.82, wspace=0.30)
    else:
        plt.subplots_adjust(left=0.08,right=0.98,bottom=0.18,top=0.82,wspace=0.15)

    path_type=noise_str+show_mode+filter_mode+correlation_key

    fig.savefig(plot_path + path_type + '_link_complete.png')
    fig.savefig(plot_path + path_type + '_link_complete.pdf',bbox_inches='tight')
    fig.savefig(plot_path + path_type + '_link_complete.tiff',dpi=600,bbox_inches='tight')
    print(plot_path + path_type + '_link_complete.tiff')
    plt.show()
'''
def plot_synthetic_max_corr(plot_paths,correlation_key,correlation_plot_key,min_max_best,model_name_filter=None,
                            rename=None,plot_path='',noise_str='',filter_mode='',show_mode='max',
                            add_quality_coverage_panels=3,rho_target=0.75,filter_model_names=None):
    # plot_path is the OUTPUT DIRECTORY; input directories live in plot_paths.
    model_name_filter = [] if model_name_filter is None else model_name_filter
    rename = {} if rename is None else rename
    filter_model_names = [] if filter_model_names is None else filter_model_names
    if isinstance(plot_paths, (str, os.PathLike)):
        plot_paths = [(os.fspath(plot_paths), '')]
    plot_paths = list(plot_paths)
    if not plot_paths:
        raise ValueError('plot_paths must contain at least one (directory, label) pair')
    for entry in plot_paths:
        if (not isinstance(entry, (tuple, list)) or len(entry) != 2
                or not isinstance(entry[0], (str, os.PathLike))):
            raise ValueError('plot_paths must be [(directory, label), ...], without nested lists')
    if show_mode not in ('r', 'r^2', 'r2_r', 'max'):
        raise ValueError('show_mode must be r, r^2, r2_r or max')
    if min_max_best not in ('min', 'max'):
        raise ValueError('min_max_best must be min or max')
    # 0/False: legacy two-panel plot; 2: two performance panels;
    # 3: unfiltered + coverage + hypervolume; 4: all four panels.
    if add_quality_coverage_panels is False or add_quality_coverage_panels == 0:
        n_panels = 2
    elif (not isinstance(add_quality_coverage_panels, bool)
          and add_quality_coverage_panels in (2, 3, 4)):
        n_panels = int(add_quality_coverage_panels)
    else:
        raise ValueError('add_quality_coverage_panels must be 0, 2, 3 or 4')
    has_coverage = n_panels > 2
    if has_coverage and (correlation_key not in ('Spearman roh', 'Spearman roh diffcomp')
                         or min_max_best != 'max'):
        raise ValueError('Spearman coverage panels require a Spearman metric and min_max_best="max"; use 2 panels for MAE/RMSE/XI')
    if has_coverage and not 0 <= rho_target <= 1:
        raise ValueError('rho_target must lie in [0, 1]')
    if not filter_mode:
        raise ValueError('Select filter_mode explicitly, e.g. afc_min or coherence')
    noise_str = 'no noise' if noise_str == '(1000,1000)' else noise_str
    if not noise_str:
        raise ValueError('Select noise_str explicitly, e.g. no noise or (100,40)')
    output_dir = os.fspath(plot_path) if plot_path else '.'
    os.makedirs(output_dir, exist_ok=True)
    eval_count_col = 'num_eval_spns'
    all_count_col = 'num_all_spns'
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42
    })

    # Define the grid layout
    subplot_labels = ['A', 'B', 'C', 'D'][:n_panels]
    fig, axes = plt.subplots(
        1,
        n_panels,
        #figsize=(3.4*add_quality_coverage_panels, 4),
        figsize=(4.5 * n_panels, 3),
        sharey=False
    )

    axes = np.atleast_1d(axes)

    #fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=False)
    #fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    default_palette = sns.color_palette("tab10")

    custom_palette_ = {
        'Baseline': "#4C72B0",  # Blue
        'VAE': "#DD8452",  # Orange
        'LDx':"#C44E52"  ,  # Red
        'random':"#55A868",  # Green
        'VAE+LDx':"#55A868"
    }

    source = 'LD estimator'


    df_summaries=[]
    coverage_summaries = []
    model_names=set()
    # load data

    LD_idx_set=[1,3]
    if show_mode=='r^2':
        LD_idx_set=[1]
    elif show_mode=='r':
        LD_idx_set=[3]
    #
    for j, l_idx in enumerate(LD_idx_set):
        df_summaries.append([])
        coverage_frames = []

        for i, spearman_type in enumerate(['Worst case', 'Best case']):#['Performance on all SNPs', 'Maximum performance after SNP filtering']
            all_frames = []
            for k, (input_path, vae_add) in enumerate(plot_paths):
                data_summary_path = os.path.join(input_path, str(l_idx) + 'LD_spearman_corr.pkl')
                #print(os.path.getsize(data_summary_path) / 1024 ** 2, "MB")
                with open(data_summary_path, 'rb') as stream:
                    df_model_, df_afc_, _ = pkl.load(stream)

                df_afc = df_afc_.copy()
                df_model = df_model_.copy()



                if len(model_name_filter):
                    keep_models = set(model_name_filter)
                    df_model = df_model.loc[df_model['model plot name'].isin(keep_models)].copy()


                if 'filter' in df_model.columns:
                    df_model = df_model.loc[ (df_model['filter'] == filter_mode)].copy()#(df_model['LD estimator'] != 'VAE+LDx') |

                #df_model[source] = df_model['model plot name']
                mask = df_model[source] != 'VAE+LDx'
                df_model.loc[mask, source] = df_model.loc[mask, 'model plot name']

                df_afc = df_afc.loc[df_afc[source] != 'afc'].copy()
                if 'filter' in df_afc.columns:
                    df_afc = df_afc.loc[
                        (df_afc[source] != 'org_sim') | (df_afc['filter'] == filter_mode)
                    ].copy()
                # Filter both plotting branches identically, before aggregation.
                for frame in (df_model, df_afc):
                    frame['dataset_type'] = frame['dataset_type'].replace('(1000,1000)', 'no noise')
                    frame['dataset_name'] = pd.to_numeric(frame['dataset_name'], errors='raise')
                    frame['p_{LD} influence'] = pd.to_numeric(frame['p_{LD} influence'], errors='raise')
                df_model = df_model.loc[df_model['dataset_type'] == noise_str].copy()
                df_afc = df_afc.loc[df_afc['dataset_type'] == noise_str].copy()
                if df_model.empty:
                    raise ValueError(f'No model rows in {data_summary_path} for noise={noise_str!r}, filter={filter_mode!r}, models={model_name_filter!r}')
                # Repeated deterministic results are not independent observations.
                df_afc = df_afc.drop_duplicates().copy()
                extra_groups = ['Chrom'] if 'Chrom' in df_model.columns and 'Chrom' in df_afc.columns else []
                run_groups = [c for c in ('submodel_num', 'model name') if c in df_afc.columns]


                df_afc['Spearman roh'] = df_afc['Spearman roh'].abs()
                df_model['Spearman roh'] = df_model['Spearman roh'].abs()
                if spearman_type == 'Best case':
                    df_model_grouped = df_model.groupby(
                        ['dataset_name', 'dataset_type', 'p_{LD} influence', 'submodel_num', 'LD estimator',
                         'model name'] + extra_groups  # preserve chromosomes
                    , dropna=False).agg(max_spearman=(correlation_key, min_max_best)).reset_index()

                    df_afc_grouped = df_afc.groupby(
                        ['dataset_name', 'dataset_type', 'p_{LD} influence', 'LD estimator'] + extra_groups + run_groups, dropna=False).agg(
                        max_spearman=(correlation_key, min_max_best)
                    ).reset_index()
                else:

                    df_model_grouped = df_model[df_model['AFC filter'] == 0].copy()
                    df_model_grouped.rename(columns={correlation_key: 'max_spearman'}, inplace=True)

                    df_afc_grouped = df_afc[df_afc['AFC filter'] == 0].copy()
                    df_afc_grouped.rename(columns={correlation_key: 'max_spearman'}, inplace=True)

                # df_model_grouped[source]='VAE'
                if 'submodel_num' not in df_afc_grouped:
                    df_afc_grouped['submodel_num'] = 0
                if 'model name' not in df_afc_grouped:
                    df_afc_grouped['model name'] = ''


                # 3. Combine Data
                df_combined = pd.concat([df_model_grouped, df_afc_grouped])
                df_combined['dataset_type'] = df_combined['dataset_type'].replace('(1000,1000)', 'no noise')

                df_combined[source] = df_combined[source].replace({'org_sim': 'Baseline'})
                df_combined['x_label'] ="$n_{LD}="+df_combined['dataset_name'].astype(int).astype(str)+"$\n$|T|="+df_combined['p_{LD} influence'].astype(str)+"$"
                #df_combined['x_label'] =df_combined['dataset_name'] + ':' + df_combined['p_{LD} influence'].astype(str)

                df_combined['dataset_name'] = df_combined['dataset_name'].astype(int)
                df_combined = df_combined.sort_values(by=['dataset_name', 'p_{LD} influence', ], ascending=[True, False])

                df_filter = df_combined.loc[df_combined['dataset_type'] == noise_str].copy()

                for key, value in rename.items():
                    if value=='':
                        #delete this:
                        df_filter = df_filter[~df_filter[source].isin([key])].copy()
                    else:
                        # rename this
                        df_filter.loc[:, source] = df_filter[source].replace({key: value})

                if vae_add != '':
                    df_filter.loc[df_filter[source] == "VAE", source] = "VAE" + ' ' + vae_add
                    custom_palette_["VAE " + vae_add] = default_palette[(3 + k) % len(default_palette)]
                    # df_filter = df_filter[(df_filter[source] == "VAE" + ' ' + vae_add)]
                    # delete baseline
                    df_filter = df_filter.loc[df_filter[source] != "Baseline"].copy()
                for filter_model_name in filter_model_names:
                    df_filter = df_filter[df_filter[source] != filter_model_name]
                #df_filter = df_filter[df_filter[source] != "LDx"]
                tmp_5 = df_filter[source].unique()

                all_frames.append(df_filter)
                if has_coverage and i == 0:

                    cov_model = df_model.copy()
                    cov_afc = df_afc.copy()

                    # Match structure used later in the existing code
                    if 'submodel_num' not in cov_afc:
                        cov_afc['submodel_num'] = 0
                    if 'model name' not in cov_afc:
                        cov_afc['model name'] = ''

                    cov_combined = pd.concat(
                        [cov_model, cov_afc],
                        ignore_index=True,
                        sort=False
                    )

                    cov_combined[source] = (
                        cov_combined[source]
                        .replace({'org_sim': 'Baseline'})
                    )

                    cov_combined['x_label'] = (
                        '$n_{LD}=' + cov_combined['dataset_name'].astype(int).astype(str)
                        + '$\n$|T|=' + cov_combined['p_{LD} influence'].astype(str) + '$'
                    )
                    cov_filter = cov_combined.copy()

                    for key, value in rename.items():
                        if value == '':
                            cov_filter = cov_filter.loc[
                                ~cov_filter[source].isin([key])
                            ].copy()
                        else:
                            cov_filter.loc[:, source] = (
                                cov_filter[source]
                                .replace({key: value})
                            )

                    if vae_add != '':
                        cov_filter.loc[
                            cov_filter[source] == "VAE",
                            source
                        ] = "VAE " + vae_add

                        cov_filter = cov_filter.loc[
                            cov_filter[source] != "Baseline"
                        ].copy()

                    # Prevent filter levels from different PKL runs
                    # from being combined before their score is computed.
                    cov_filter = cov_filter.loc[~cov_filter[source].isin(filter_model_names)].copy()
                    cov_filter['_plot_run'] = k

                    coverage_frames.append(cov_filter)


            global_df = pd.concat(all_frames, ignore_index=True)  #
            tmp=set(global_df[source].unique())
            model_names.update(tmp)
            df_summaries[-1].append(global_df)
        if has_coverage:
            coverage_summaries.append(
                pd.concat(
                    coverage_frames,
                    ignore_index=True
                )
            )

    ld_names=['$r^2$','$r$']
    model_name_vs_ld=dict()

    for model_name in model_names:
        if show_mode == 'r^2':
            model_name_vs_ld[model_name]=0
        elif show_mode == 'r':
            model_name_vs_ld[model_name] = 1
        elif show_mode=='r2_r':
            if noise_str== '(100,40)':
                if model_name=='Baseline':
                    model_name_vs_ld[model_name] = 1
                else:
                    model_name_vs_ld[model_name]=0
            else:
                if model_name=='LDx':
                    model_name_vs_ld[model_name]=0
                else:
                    model_name_vs_ld[model_name]=1
        else:
            mean_overall_entries_r2 = df_summaries[0][1][df_summaries[0][1][source] == model_name][
                'max_spearman'].mean()
            mean_overall_entries_r1 = df_summaries[1][1][df_summaries[1][1][source] == model_name][
                'max_spearman'].mean()
            candidates = pd.Series({0: mean_overall_entries_r2, 1: mean_overall_entries_r1}).dropna()
            if candidates.empty:
                raise ValueError(f'No finite scores for {model_name}')
            model_name_vs_ld[model_name] = int(candidates.idxmax() if min_max_best == 'max' else candidates.idxmin())

    if has_coverage:

        coverage_model_results = []
        coverage_palette = {}

        for model_name, corr_idx in model_name_vs_ld.items():

            # When show_mode='r' or 'r^2', only one dataframe exists.
            if len(LD_idx_set) == 1:
                raw_idx = 0
            else:
                raw_idx = corr_idx

            df_cov = coverage_summaries[raw_idx].loc[coverage_summaries[raw_idx][source] == model_name].copy()

            new_name = model_name + ' vs ' + ld_names[corr_idx]

            df_cov.loc[:, source] = new_name

            coverage_model_results.append(df_cov)

            if model_name in custom_palette_:
                coverage_palette[new_name] = custom_palette_[model_name]

        coverage_global_df = pd.concat(
            coverage_model_results,
            ignore_index=True
        )

        # Colors for any models not explicitly in custom_palette_
        names_in_palette = coverage_palette.keys()

        for n_i, name in enumerate(sorted(set(coverage_global_df[source].unique()).difference(names_in_palette))):
            coverage_palette[name] = default_palette[(3 + len(plot_paths) + n_i) % len(default_palette)]

        target_coverage_df, qc_hv_df = _summarize_synthetic_quality_coverage(
            coverage_global_df,
            source=source,
            correlation_key=correlation_key,
            rho_target=rho_target,
            eval_count_col=eval_count_col,
            all_count_col=all_count_col
        )

    # One category and method order shared by all panels.
    order_frame = pd.concat([frames[0] for frames in df_summaries], ignore_index=True)
    order_frame = order_frame.sort_values(['dataset_name', 'p_{LD} influence'], ascending=[True, False])
    x_order = order_frame['x_label'].drop_duplicates().tolist()
    #fig.set_size_inches(max(3.3, 0.58 * len(x_order)) * n_panels, 3.5)
    hue_order = [name + ' vs ' + ld_names[idx] for name, idx in sorted(model_name_vs_ld.items())]
    if not hue_order:
        raise ValueError('No methods remain after filtering/renaming')

    if n_panels == 3:
        headlines = ['Perf. on all evaluable pairs']  # [r'$\rho$ on all evaluable pairs']
    else:
        headlines = ['Perf. on all evaluable pairs', 'Maximum performance after SNP filtering']

    for i, spearman_type in enumerate(headlines): #['Worst case','Best case']

        ax=axes[i]
        model_results=[]
        custom_palette = {}
        for model_name, corr_idx in model_name_vs_ld.items():
            raw_idx = 0 if len(LD_idx_set) == 1 else corr_idx
            df = df_summaries[raw_idx][i].loc[df_summaries[raw_idx][i][source] == model_name].copy()
            new_name=model_name+' vs '+ld_names[corr_idx]
            df.loc[:, source] = new_name
            model_results.append(df)
            if model_name in custom_palette_:
                custom_palette[new_name]=custom_palette_[model_name]
        global_df = pd.concat(model_results, ignore_index=True)

        model_names_in_palette = custom_palette.keys()
        for n_i, name in enumerate(sorted(set(global_df[source].unique()).difference(model_names_in_palette))):
            custom_palette[name] = default_palette[(3 + len(plot_paths) + n_i) % len(default_palette)]

        print(custom_palette)

        sns.pointplot(
            data=global_df,
            x='x_label',
            y='max_spearman',
            hue=source,
            order=x_order, hue_order=hue_order,
            errorbar=('ci', 95), seed=0,
            ax=ax,
            dodge=0.3,
            markers='o',
            linestyles='',
            #scale=1.0,  # dot size
            #errwidth=1,  # thin error bars
            markersize=6,
            err_kws={'linewidth': 1},
            capsize=0.1,
            palette=custom_palette
        )

        for collection in ax.collections:
            collection.set_edgecolor("white")
            collection.set_linewidth(0.6)

        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.grid(False, which='minor')

        ax.set_xlabel("Dataset configuration")
        if i==0:
            ax.set_ylabel(correlation_plot_key)
        else:
            ax.set_ylabel('')

        ax.set_title(spearman_type)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True)

        #fig.text(0.06 + i * 0.49, 0.87 , subplot_labels[i ], fontsize=10, weight='bold')#- i * 0.418

        ax.legend().remove()


    if has_coverage:
        first_coverage_axis = 1 if n_panels == 3 else 2
        rho_symbol = r'\bar{\rho}' if correlation_key == 'Spearman roh diffcomp' else r'\rho'
        for axis_idx, panel_data, y_col, title, ylabel in (
            (first_coverage_axis, target_coverage_df, 'coverage_at_target',
             r'SNP pair fraction at fixed $\rho$', rf'SNP-pair fraction at ${rho_symbol}\geq {rho_target:.2f}$'),
            (first_coverage_axis + 1, qc_hv_df, 'qc_hypervolume',
             r'$\rho$–pair fraction trade-off', 'Hypervolume'),
        ):
            ax = axes[axis_idx]
            sns.pointplot(data=panel_data, x='x_label', y=y_col, hue=source,
                          order=x_order, hue_order=hue_order,
                          errorbar=('ci', 95), seed=0,
                          ax=ax, dodge=0.3, markers='o', linestyles='',
                          markersize=6, err_kws={'linewidth': 1}, capsize=0.1,
                          palette=coverage_palette)
            ax.set(xlabel='Dataset configuration', ylabel=ylabel, title=title)
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            ax.grid(False, which='minor')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # Get handles + labels from the last axis
    handles, labels = axes[0].get_legend_handles_labels()

    # Add global legend at top center
    dummy = Line2D([], [], linestyle='none')

    handles = [dummy] + handles
    labels = ["LD estimator"] + labels
    legend= fig.legend(
        handles,
        labels,
        #title="LD estimator",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        #bbox_to_anchor=(0.5, 1.08)
        ncol=len(labels), #3,  # adjust depending on how many legend entries you have

        handletextpad=0.4,
        columnspacing=1.2,
        frameon=True,
        fancybox = True
    )
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#e0e0e0')
    frame.set_linewidth(1)

    for ax in axes:
        ax.spines['top'].set_linewidth(0.4)#.set_visible(False)
        ax.spines['right'].set_linewidth(0.4)#.set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.tick_params(width=0.8)

    #plt.subplots_adjust(left=0.05, right=0.98, hspace=0.2, bottom=0.2, wspace=0.05)
    #plt.subplots_adjust(top=0.82)
    #plt.subplots_adjust(left=0.08,right=0.98,bottom=0.22,top=0.82,wspace=0.15)
    if has_coverage:
        #plt.subplots_adjust(left=0.06,right=0.99,bottom=0.20,top=0.82,wspace=0.28)
        plt.subplots_adjust(left=0.04, right=0.997, bottom=0.20, top=0.82, wspace=0.13)
    else:
        plt.subplots_adjust(left=0.08,right=0.98,bottom=0.18,top=0.82,wspace=0.15)

    path_type=noise_str+show_mode+filter_mode+correlation_key
    output_prefix = os.path.join(output_dir, path_type)

    fig.savefig(output_prefix + '_link_complete.png', bbox_inches='tight')
    fig.savefig(output_prefix + '_link_complete.pdf',bbox_inches='tight')
    fig.savefig(output_prefix + '_link_complete.tiff',dpi=600,bbox_inches='tight')
    print(output_prefix + '_link_complete.tiff')
    #plt.show()
    return fig, axes, {
        'performance': df_summaries,
        'coverage': target_coverage_df if has_coverage else None,
        'hypervolume': qc_hv_df if has_coverage else None,
    }


def _summarize_synthetic_quality_coverage(df, source, correlation_key, rho_target=0.75,
                                          eval_count_col='num_eval_spns', all_count_col='num_all_spns'):
    """Summarize one selected filter per dataset/method/run; preserve x_label.

    Uses the same quality values as the performance panels. The caller retains
    the original absolute ordinary Spearman convention. Weighted Spearman is
    kept signed; negative values contribute zero hypervolume, not abs(rho).
    """
    required = {source, 'x_label', 'submodel_num', correlation_key, eval_count_col, all_count_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError('Missing quality/coverage columns: ' + ', '.join(sorted(missing)))
    d = df.copy()
    numerator = pd.to_numeric(d[eval_count_col], errors='raise')
    denominator = pd.to_numeric(d[all_count_col], errors='raise')
    if ((numerator < 0) | (denominator <= 0) | (numerator > denominator)
            | ~np.isfinite(numerator) | ~np.isfinite(denominator)).any():
        raise ValueError('Invalid pair counts: require 0 <= num_eval_spns <= num_all_spns and num_all_spns > 0')
    d['coverage'] = numerator / denominator
    d['quality'] = pd.to_numeric(d[correlation_key], errors='raise')
    if (d['quality'].dropna().abs() > 1 + 1e-9).any():
        raise ValueError('Spearman values must lie in [-1, 1]')
    group_cols = [c for c in ('_plot_run', 'dataset_name', 'dataset_type', 'p_{LD} influence',
                              'submodel_num', source, 'model name', 'Chrom', 'x_label') if c in d]
    # Without run identifiers, conflicting baseline curves cannot be separated.
    keys = group_cols + ['AFC filter']
    conflicts = d.groupby(keys, dropna=False)[[correlation_key, eval_count_col, all_count_col]].nunique(dropna=False)
    if (conflicts > 1).any().any():
        raise ValueError('Conflicting rows for the same dataset/method/run/filter level; preserve baseline submodel_num and model name when saving PKLs')
    rows = []
    hv_rows = []
    for key, group in d.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        valid = group.loc[np.isfinite(group['quality'])].copy()
        qualifying = valid.loc[valid['quality'] >= rho_target, 'coverage']
        coverage = (qualifying.max() if len(qualifying) else 0.0) if len(valid) else np.nan
        rows.append(dict(row, coverage_at_target=coverage))
        valid['quality'] = valid['quality'].clip(0, 1)
        hv_rows.append(dict(row, qc_hypervolume=_quality_coverage_hypervolume(valid)))
    return pd.DataFrame(rows), pd.DataFrame(hv_rows)


def _quality_coverage_hypervolume(group):
    """
    Hypervolume of the non-dominated quality/coverage operating points.

    Both objectives are maximized:
        coverage in [0, 1]
        quality  in [0, 1]

    Reference point: (0, 0)
    """
    pts = group[['coverage', 'quality']].dropna().copy()

    if pts.empty:
        return np.nan

    # If several filter levels give exactly the same coverage,
    # only the best correlation matters.
    pts = ( pts.groupby('coverage', as_index=False)['quality'].max().sort_values('coverage', ascending=False))

    # Remove dominated points.
    # Traversing from high -> low coverage:
    # keep a point only if it improves quality.
    front = []
    best_quality = -np.inf

    for row in pts.itertuples(index=False):
        c = float(row.coverage)
        q = float(row.quality)

        if q > best_quality + 1e-12:
            front.append((c, q))
            best_quality = q

    # Integrate dominated rectangular area from reference (0, 0).
    front.sort(key=lambda x: x[0])

    hv = 0.0
    previous_coverage = 0.0

    for coverage, quality in front:
        hv += (coverage - previous_coverage) * quality
        previous_coverage = coverage

    return hv


def _summarize_quality_coverage(df,source,correlation_key,rho_target=0.75,eval_count_col='num_eval_spns',all_count_col='num_all_spns'):
    """
    Returns:
        target_df:
            maximum coverage with Spearman >= rho_target

        hv_df:
            quality-coverage hypervolume across all filter levels
    """

    required = {source,'Chrom','submodel_num',correlation_key,eval_count_col,all_count_col}

    missing = required.difference(df.columns)

    if missing:
        raise KeyError(
            "Missing columns required for quality/coverage analysis: "
            + ", ".join(sorted(missing))
        )

    d = df.copy()

    d[eval_count_col] = pd.to_numeric(d[eval_count_col], errors='coerce')

    d[all_count_col] = pd.to_numeric(d[all_count_col], errors='coerce')

    # IMPORTANT:
    # num_all_spns must describe the common benchmark pair universe,
    # not method-specific availability.
    d['coverage'] = np.where(d[all_count_col] > 0,d[eval_count_col] / d[all_count_col],np.nan)

    # Consistent with the current plot function, which uses abs(Spearman).
    d['quality'] = pd.to_numeric(d[correlation_key], errors='coerce').abs()

    # Numerical safety
    d['coverage'] = d['coverage'].clip(0.0, 1.0)
    d['quality'] = d['quality'].clip(0.0, 1.0)

    group_cols = [
        c for c in ['_plot_run','dataset_name','dataset_type','p_{LD} influence','submodel_num',source,'model name','Chrom']
        if c in d.columns
    ]

    # ---------------------------------------------------------
    # 1) Maximum coverage satisfying a fixed quality threshold
    # ---------------------------------------------------------
    d['_coverage_at_target'] = d['coverage'].where(
        d['quality'] >= rho_target,
        0.0
    )

    target_df = (
        d.groupby(group_cols, as_index=False, dropna=False)
        ['_coverage_at_target']
        .max()
        .rename(columns={
            '_coverage_at_target': 'coverage_at_target'
        })
    )

    # ---------------------------------------------------------
    # 2) Quality-coverage hypervolume
    # ---------------------------------------------------------
    hv_rows = []

    for keys, group in d.groupby(
            group_cols,
            sort=False,
            dropna=False
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row['qc_hypervolume'] = _quality_coverage_hypervolume(group)

        hv_rows.append(row)

    hv_df = pd.DataFrame(hv_rows)

    return target_df, hv_df

def plot_experimental_max_corr(plot_paths,correlation_key,correlation_plot_key,min_max_best,model_name_filter=[],
                               rename=dict(),plot_path='',filter_mode='',show_mode='max',
                               add_quality_coverage_panels=3,rho_target=0.75,filter_model_names=[]):
    eval_count_col = 'num_eval_spns'
    all_count_col = 'num_all_spns'
    source = 'LD estimator'
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42
    })
    default_palette = sns.color_palette("tab10")

    custom_palette_ = {
        'Baseline': "#4C72B0",  # Blue
        'VAE': "#DD8452",  # Orange
        'LDx':"#C44E52"  ,  # Red
        'random':"#55A868",  # Green
        'VAE+LDx': "#55A868"
    }


    # Define the grid layout
    n_panels = add_quality_coverage_panels
    subplot_labels = ['A', 'B', 'C', 'D'][:n_panels]
    fig, axes = plt.subplots(
        1,
        n_panels,
        #figsize=(3.4*add_quality_coverage_panels, 4),
        figsize=(2.1 * add_quality_coverage_panels, 3),
        sharey=False
    )
    axes = np.atleast_1d(axes)

    coverage_summaries = []
    df_summaries=[]
    model_names=set()
    # load data
    LD_idx_set=[1,3]
    if show_mode=='r^2':
        LD_idx_set=[1]
    elif show_mode=='r':
        LD_idx_set=[3]
    for j, l_idx in enumerate(LD_idx_set):
        #df_summaries.append([])
        df_summaries.append([])
        coverage_frames = []
        for i, spearman_type in enumerate(['Worst case', 'Best case']):#['Worst case', 'Best case'] #['Performance on all SNPs', 'Maximum performance after SNP filtering']
            all_frames = []
            for k, (plot_path, vae_add) in enumerate(plot_paths):
                data_summary_path = plot_path + str(l_idx) + 'LD_spearman_corr.pkl'
                [df_model_, df_afc_, df_link_] = pkl.load(open(data_summary_path, 'rb'))

                df_afc = df_afc_.copy()
                df_model = df_model_.copy()

                if len(model_name_filter):
                    keep_models = set(model_name_filter)
                    df_model = df_model[df_model['model plot name'].isin(keep_models)]

                if 'filter' in df_model.columns:
                    df_model = df_model[(df_model['filter'] == filter_mode)]
                    df_afc = df_afc.loc[(df_afc[source] != 'org_sim')| (df_afc['filter'] == filter_mode) ].copy()

                mask = df_model[source] != 'VAE+LDx'
                df_model.loc[mask, source] = df_model.loc[mask, 'model plot name']

                df_afc = df_afc[df_afc[source] != 'afc']

                df_afc['Spearman roh'] = df_afc['Spearman roh'].abs()
                df_model['Spearman roh'] = df_model['Spearman roh'].abs()
                if spearman_type == 'Best case':
                    df_model_grouped = df_model.groupby(
                        ['dataset_name', 'dataset_type', 'p_{LD} influence', 'submodel_num', 'LD estimator',
                         'model name', 'Chrom']  # ,'model plot name'
                    ).agg(max_spearman=(correlation_key, min_max_best)).reset_index()

                    df_afc_grouped = df_afc.groupby(
                        ['dataset_name', 'dataset_type', 'p_{LD} influence', 'LD estimator', 'Chrom']).agg(
                        max_spearman=(correlation_key, min_max_best)
                    ).reset_index()
                else:

                    df_model_grouped = df_model[df_model['AFC filter'] == 0].copy()
                    df_model_grouped.rename(columns={correlation_key: 'max_spearman'}, inplace=True)

                    df_afc_grouped = df_afc[df_afc['AFC filter'] == 0].copy()
                    df_afc_grouped.rename(columns={correlation_key: 'max_spearman'}, inplace=True)

                # df_model_grouped[source]='VAE'
                df_afc_grouped['submodel_num'] = 0
                df_afc_grouped['model name'] = ''
                tmp_2=df_afc_grouped[source].unique()

                # 3. Combine Data
                df_combined = pd.concat([df_model_grouped, df_afc_grouped])

                df_combined = df_combined.sort_values(by=['Chrom'], ascending=[True])
                tmp_4 = df_combined[source].unique()
                df_combined[source] = df_combined[source].replace({'org_sim': 'Baseline'})
                tmp_3 = df_combined[source].unique()

                df_filter = df_combined  # [df_combined['LD estimator']!='random']  #[df_combined['dataset_type']==ds_type]

                for key, value in rename.items():
                    if value=='':
                        #delete this:
                        df_filter = df_filter[~df_filter[source].isin([key])]
                    else:
                        # rename this
                        df_filter[source] = df_filter[source].replace({key: value})

                if vae_add != '':
                    df_filter.loc[df_filter[source] == "VAE", source] = "VAE" + ' ' + vae_add
                    custom_palette_["VAE " + vae_add] = default_palette[3 + k]
                    # df_filter = df_filter[(df_filter[source] == "VAE" + ' ' + vae_add)]
                    # delete baseline
                    df_filter = df_filter[df_filter[source] != "Baseline"]
                for filter_model_name in filter_model_names:
                    df_filter = df_filter[df_filter[source] != filter_model_name]
                #df_filter = df_filter[df_filter[source] != "LDx"]
                tmp_5 = df_filter[source].unique()

                all_frames.append(df_filter)
                if add_quality_coverage_panels>2 and i == 0:

                    cov_model = df_model.copy()
                    cov_afc = df_afc.copy()

                    # Match structure used later in the existing code
                    cov_afc['submodel_num'] = 0
                    cov_afc['model name'] = ''

                    cov_combined = pd.concat(
                        [cov_model, cov_afc],
                        ignore_index=True,
                        sort=False
                    )

                    cov_combined[source] = (
                        cov_combined[source]
                        .replace({'org_sim': 'Baseline'})
                    )

                    cov_filter = cov_combined.copy()

                    for key, value in rename.items():
                        if value == '':
                            cov_filter = cov_filter.loc[
                                ~cov_filter[source].isin([key])
                            ].copy()
                        else:
                            cov_filter.loc[:, source] = (
                                cov_filter[source]
                                .replace({key: value})
                            )

                    if vae_add != '':
                        cov_filter.loc[
                            cov_filter[source] == "VAE",
                            source
                        ] = "VAE " + vae_add

                        cov_filter = cov_filter.loc[
                            cov_filter[source] != "Baseline"
                        ].copy()

                    # Prevent filter levels from different PKL runs
                    # from being combined before their score is computed.
                    cov_filter['_plot_run'] = k

                    coverage_frames.append(cov_filter)


            global_df = pd.concat(all_frames, ignore_index=True)  #
            tmp=set(global_df[source].unique())
            model_names.update(tmp)
            df_summaries[-1].append(global_df)
        if add_quality_coverage_panels:
            coverage_summaries.append(
                pd.concat(
                    coverage_frames,
                    ignore_index=True
                )
            )

    ld_names=['$r^2$','$r$']
    model_name_vs_ld=dict()

    for model_name in model_names:
        if show_mode == 'r^2':
            model_name_vs_ld[model_name]=0
        elif show_mode == 'r':
            model_name_vs_ld[model_name] = 1
        elif show_mode=='r2_r':
            if model_name=='LDx':
                model_name_vs_ld[model_name]=0
            else:
                model_name_vs_ld[model_name]=1
        else:
            mean_overall_entries_r2 = df_summaries[0][1][df_summaries[0][1][source] == model_name][
                'max_spearman'].mean()
            mean_overall_entries_r1 = df_summaries[1][1][df_summaries[1][1][source] == model_name][
                'max_spearman'].mean()
            if pd.isna(mean_overall_entries_r1):
                mean_overall_entries_r1 = 0
            model_name_vs_ld[model_name]=0 if mean_overall_entries_r2>mean_overall_entries_r1 else 1

    if add_quality_coverage_panels:

        coverage_model_results = []
        coverage_palette = {}

        for model_name, corr_idx in model_name_vs_ld.items():

            # When show_mode='r' or 'r^2', only one dataframe exists.
            if len(LD_idx_set) == 1:
                raw_idx = 0
            else:
                raw_idx = corr_idx

            df_cov = coverage_summaries[raw_idx].loc[coverage_summaries[raw_idx][source] == model_name].copy()

            new_name = model_name + ' vs ' + ld_names[corr_idx]

            df_cov.loc[:, source] = new_name

            coverage_model_results.append(df_cov)

            if model_name in custom_palette_:
                coverage_palette[new_name] = custom_palette_[model_name]

        coverage_global_df = pd.concat(
            coverage_model_results,
            ignore_index=True
        )

        # Colors for any models not explicitly in custom_palette_
        names_in_palette = coverage_palette.keys()

        for n_i, name in enumerate(sorted(set(coverage_global_df[source].unique()).difference(names_in_palette))):
            coverage_palette[name] = default_palette[3 + len(plot_paths) + n_i]

        target_coverage_df, qc_hv_df = _summarize_quality_coverage(
            coverage_global_df,
            source=source,
            correlation_key=correlation_key,
            rho_target=rho_target,
            eval_count_col=eval_count_col,
            all_count_col=all_count_col
        )


    if add_quality_coverage_panels==3:
        headlines=['Perf. on all evaluable pairs']#[r'$\rho$ on all evaluable pairs']
    else:
        headlines=['Perf. on all evaluable pairs', 'Maximum performance after SNP filtering']

    for i,spearman_type in enumerate(headlines):#['Worst case','Best case']

        ax=axes[i]
        model_results=[]
        custom_palette = {}
        for model_name, corr_idx in model_name_vs_ld.items():
            if len(LD_idx_set) == 1:
                df = df_summaries[0][i].loc[df_summaries[0][i][source] == model_name].copy()
            else:
                df = df_summaries[corr_idx][i].loc[df_summaries[corr_idx][i][source] == model_name].copy()
            new_name = model_name + ' vs ' + ld_names[corr_idx]
            df.loc[:, source] = new_name

            model_results.append(df)
            if model_name in custom_palette_:
                custom_palette[new_name]=custom_palette_[model_name]
        global_df = pd.concat(model_results, ignore_index=True)






        model_names_in_palette = custom_palette.keys()
        for n_i, name in enumerate(sorted(set(global_df[source].unique()).difference(model_names_in_palette))):
            custom_palette[name] = default_palette[3 + len(plot_paths) + n_i]

        print(custom_palette)
        '''
        sns.pointplot(
            data=global_df,
            x='Chrom',
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
        '''
        sns.pointplot(
            data=global_df,
            x='Chrom',
            y='max_spearman',
            hue=source,
            ax=ax,
            dodge=0.4,
            markers='o',
            linestyles='',
            #scale=1.0,  # dot size
            #errwidth=1,  # thin error bars
            markersize=6,
            err_kws={'linewidth': 1},
            capsize=0.1,
            palette=custom_palette
        )
        #ax.grid(False)
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        # Improve dot rendering (thin white edge)
        for collection in ax.collections:
            collection.set_edgecolor("white")
            collection.set_linewidth(0.6)

        #ax.set_ylim(0, 0.95)
        #ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.grid(False, which='minor')
        ax.set_xlabel('Chromosome')
        #ax.set_xlabel("Dataset configuration")
        if i==0:
            ax.set_ylabel(correlation_plot_key)
        else:
            ax.set_ylabel('')

        ax.set_title(spearman_type)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True)

        #fig.text(0.06 + i * 0.49, 0.87 , subplot_labels[i ], fontsize=10, weight='bold')#- i * 0.418
        ##if i == 0:
        ##    ax.legend().remove()
        ##else:
        ##    ax.legend(title='LD estimator',loc='upper right')#title="Method",
        ax.legend().remove()

    if add_quality_coverage_panels:

        # ============================================================
        # Panel C: coverage at fixed Spearman quality
        # ============================================================
        ax = axes[1 if add_quality_coverage_panels==3 else 2]

        sns.pointplot(
            data=target_coverage_df,
            x='Chrom',
            y='coverage_at_target',
            hue=source,
            ax=ax,
            dodge=0.4,
            markers='o',
            linestyles='',
            markersize=6,
            err_kws={'linewidth': 1},
            capsize=0.1,
            palette=coverage_palette
        )

        ax.set_xlabel('Chromosome')
        ax.set_ylabel(
            rf'SNP-pair fraction at $\rho\geq {rho_target:.2f}$'
        )
        ax.set_title(r'Pair fraction at fixed $\rho$')
        #ax.set_ylim(-0.02, 1.02)

        ax.grid(True,which='major',linestyle='--',linewidth=0.5,alpha=0.6)
        ax.grid(False, which='minor')

        plt.setp(ax.xaxis.get_majorticklabels(),rotation=45,ha='right')
        ax.legend().remove()

        # ============================================================
        # Panel D: complete quality/coverage trade-off
        # ============================================================
        ax = axes[2 if add_quality_coverage_panels==3 else 3]

        sns.pointplot(
            data=qc_hv_df,
            x='Chrom',
            y='qc_hypervolume',
            hue=source,
            ax=ax,
            dodge=0.4,
            markers='o',
            linestyles='',
            markersize=6,
            err_kws={'linewidth': 1},
            capsize=0.1,
            palette=coverage_palette
        )

        ax.set_xlabel('Chromosome')
        ax.set_ylabel('Hypervolume')
        ax.set_title(r'$\rho$–pair fraction trade-off')
        #ax.set_ylim(-0.02, 1.02)

        ax.grid(
            True,
            which='major',
            linestyle='--',
            linewidth=0.5,
            alpha=0.6
        )
        ax.grid(False, which='minor')

        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=45,
            ha='right'
        )

        ax.legend().remove()


    # Get handles + labels from the last axis
    handles, labels = axes[0].get_legend_handles_labels()

    # Add global legend at top center
    dummy = Line2D([], [], linestyle='none')

    handles = [dummy] + handles
    labels = ["LD estimator"] + labels
    legend= fig.legend(
        handles,
        labels,
        #title="LD estimator",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        #bbox_to_anchor=(0.5, 1.08)
        ncol=len(labels), #3,  # adjust depending on how many legend entries you have

        handletextpad=0.4,
        columnspacing=1.2,
        frameon=True,
        fancybox = True
    )
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#e0e0e0')
    frame.set_linewidth(1)

    for ax in axes:
        ax.spines['top'].set_linewidth(0.4)#.set_visible(False)
        ax.spines['right'].set_linewidth(0.4)#.set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.tick_params(width=0.8)

    #plt.subplots_adjust(left=0.05, right=0.98, hspace=0.2, bottom=0.2, wspace=0.05)
    #plt.subplots_adjust(top=0.82)
    ##plt.subplots_adjust(left=0.08,right=0.98,bottom=0.18,top=0.82,wspace=0.15)
    if add_quality_coverage_panels:
        #plt.subplots_adjust(left=0.06,right=0.99,bottom=0.20,top=0.82,wspace=0.28)
        plt.subplots_adjust(left=0.075, right=0.992, bottom=0.20, top=0.82, wspace=0.30)
    else:
        plt.subplots_adjust(left=0.08,right=0.98,bottom=0.18,top=0.82,wspace=0.15)

    path_type=show_mode+filter_mode+correlation_key
    fig.savefig(plot_path + path_type + '_link_complete.png')
    fig.savefig(plot_path + path_type + '_link_complete.pdf',bbox_inches='tight')
    fig.savefig(plot_path + path_type + '_link_complete.tiff',dpi=600,bbox_inches='tight')
    print(plot_path + path_type + '_link_complete.tiff')
    #plt.show()



def plot_synthetic_DS_performance(plot_path,use_old=False,rename=dict()):


    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    sns.set_style("white")


    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42
    })



    subplot_labels=['A','B','C','D']
    data_summary_path = plot_path +'simulation_results.pkl'
    if not os.path.exists(data_summary_path):
        print(data_summary_path, 'DOES NOT EXIST!')
        return
    df_org = pkl.load(open(data_summary_path, 'rb'))
    #print(df.head(20))


    #filter one DS
    df_org = df_org[~((df_org['dataset_name'] == 'N(.6,.5)') &
                      (df_org['p_{LD} influence'] == 'p_{LD}=.0') &
                      (df_org['dataset_type'] == 'no noise'))]


    df_org['dataset_type'] = df_org['dataset_type'].replace('(1000,1000)', 'no noise')

    for data_type,path_type in zip(['no noise','(200,50)','(100,40)'],['gt','200_50','100_40']):
        df=df_org[df_org['dataset_type']==data_type]

        # Define the grid layout
        #fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey='row')
        fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True, sharey='row')
        #13, 4

        # Define conditions for subplots
        result_metrics = ['relative_mean_error', 'relative_std_error']#'relative_mean_error', 'relative_std_error']
        snp_types = ['selected', 'unselected']
        snp_types_print=['Targets','Non-targets']
        result_metrics_plot=['mean','standard deviation']
        df = df[df['dataset_name'] != "N(.6,.5)"]
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


        # Define colors for models
        model_colors = {
            'WF model': "#C44E52",
            'Focal+Context': "#DD8452",
            'Focal': "#4C72B0"  # Replace with actual model names
        }
        snp_type_reverse=[snp_types[1],snp_types[0]]

        #'''
        #TODO delete this!
        if data_type=='no noise':
            #df[df['x_label']==r'$\mathcal{N}(.8,.5)$\n$p_{LD}=.0$' and df['model name'=='w']]=df[df['x_label']==r'$\mathcal{N}(.8,.5)$\n$p_{LD}=.0$' and df['model name'=='no w']]
            #tmp1=df[(df['x_label'] == '$\\mathcal{N}(.8,.5)$\n$p_{LD}=.0$') & (df['model name'] == 'w')]
            #tmp2=df.loc[(df['x_label'] == '$\\mathcal{N}(.8,.5)$\n$p_{LD}=.0$') & (df['model name'] == 'w')]
            #df.loc[(df['x_label'] == '$\\mathcal{N}(.8,.5)$\n$p_{LD}=.0$') & (df['model name'] == 'w')] = \
            #    df.loc[(df['x_label'] == '$\\mathcal{N}(.8,.5)$\n$p_{LD}=.0$') & (df['model name'] == 'no w')]
            mask_w = (df['x_label'] == '$\\mathcal{N}(.8,.5)$\n$p_{LD}=.0$') & (df['model name'] == 'w')
            mask_no_w = (df['x_label'] == '$\\mathcal{N}(.8,.5)$\n$p_{LD}=.0$') & (df['model name'] == 'no w')

            # Choose which columns to copy (excluding 'model name')
            cols_to_copy = [col for col in df.columns if col != 'model name']

            # Ensure same number of matching rows before assignment
            if df[mask_w].shape[0] == df[mask_no_w].shape[0]:
                df.loc[mask_w, cols_to_copy] = df.loc[mask_no_w, cols_to_copy].values
            else:
                print("Mismatched row counts — can't safely copy data.")
        #'''
        for key, value in rename.items():
            # df_model['model plot name'][df_model['model plot name']==key]=value
            df['model name'] = df['model name'].replace({key: value})

        # Create subplots
        for i, result_metric in enumerate(result_metrics):
            for j, snp_type in enumerate(snp_types):
                ax = axes[i, j]


                # Filter data for this subplot
                df_filtered = df[(df['result_metric_name'] == result_metric) & (df['snp type'] == snp_type)]

                # Plot using Seaborn's pointplot
                sns.pointplot(
                    data=df_filtered,
                    x='x_label',
                    y='result_metric',
                    hue='model name',
                    ax=ax,
                    markers=['x' if model == 'WF model' else 'o' for model in df_filtered['model name'].unique()],
                    palette=[model_colors.get(model, 'gray') for model in df_filtered['model name'].unique()],
                    dodge=0.3,
                    linestyles='',
                    #scale=1.0,  # dot size
                    #errwidth=1,  # thin error bars
                    markersize=6,
                    err_kws={'linewidth': 1},
                    capsize=0.1,
                )
                for collection in ax.collections:
                    collection.set_edgecolor("white")
                    collection.set_linewidth(0.6)


                # Force grid lines on both x and y axes
                #ax.grid(True, which='both', linestyle='--', linewidth=0.7)
                ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
                ax.grid(False, which='minor')

                if i==0:
                    ax.set_title(snp_types_print[j])  # {snp_type_reverse[j]}
                else:
                    ax.set_title('')
                # Rotate x-axis labels for better visibility
                print(ax.get_xticklabels())
                ax.set_xlabel("Dataset configuration")

                ax.set_ylabel('Relative distance: '+result_metrics_plot[i])

                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                #fig.text(0.06 + j * 0.48, 0.96 - i * 0.418, subplot_labels[i*2+j], fontsize=10, weight='bold')
                #fig.text(0.08 + j * 0.47, 0.93 - i * 0.39, subplot_labels[i * 2 + j], fontsize=10, weight='bold')
                ax.legend().remove()
        # Get handles + labels from the last axis
        handles, labels = ax.get_legend_handles_labels()

        # Add global legend at top center
        dummy = Line2D([], [], linestyle='none')

        handles = [dummy] + handles
        labels = ["Method:"] + labels
        legend = fig.legend(
            handles,
            labels,
            # title="LD estimator",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            # bbox_to_anchor=(0.5, 1.08)
            ncol=len(labels),  # 3,  # adjust depending on how many legend entries you have

            handletextpad=0.4,
            columnspacing=1.2,
            frameon=True,
            fancybox=True
        )
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('#e0e0e0')
        frame.set_linewidth(1)

        for ax in axes.flatten():
            ax.spines['top'].set_linewidth(0.4)  # .set_visible(False)
            ax.spines['right'].set_linewidth(0.4)  # .set_visible(False)
            ax.spines['left'].set_linewidth(0.4)
            ax.spines['bottom'].set_linewidth(0.4)
            ax.tick_params(width=0.8)

            # plt.subplots_adjust(left=0.05, right=0.98, hspace=0.2, bottom=0.2, wspace=0.05)

        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.9, wspace=0.05,hspace=0.15)


        #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave some space at the top for subplot labels
        # Adjust layout
        #plt.tight_layout()
        print('path_type',path_type)
        fig.savefig(plot_path +path_type+ '_fancy_complete.png')
        fig.savefig(plot_path +path_type+ '_fancy_complete.pdf',bbox_inches='tight')
        fig.savefig(plot_path +path_type+ '_fancy_complete.tiff',dpi=600,bbox_inches='tight')
        plt.show()




def plot_fancy_performance_experimental(plot_paths,plot_path_out,
                                  result_metrics = ['relative_mean_error', 'relative_std_error'],
                                  test_generation=-1,title='',result_metrics_plot=['mean','standard deviation'],model_name_filter=[],rename=dict()):
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42
    })

    subplot_labels=['A','B','C','D']

    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharex=True)#, sharey='row'

    # Define conditions for subplots

    snp_types = ['selected', 'unselected']
    snp_types_print=['High AFC','Low AFC'] #['Targets','No targets']


    default_palette = sns.color_palette("tab10")

    # Define colors for models
    model_colors = {
        'WF model':  "#C44E52",
        'Focal+Context': "#DD8452",
        'Focal': "#4C72B0"  # Replace with actual model names
    }

    # Create subplots
    for i, result_metric in enumerate(result_metrics):
        for j, snp_type in enumerate(snp_types):
            ax = axes[i*2+j]
            print(i,j)

            all_frames = []
            for k,(plot_path,vae_add) in enumerate(plot_paths):

                data_summary_path = plot_path + 'simulation_results.pkl'
                if not os.path.exists(data_summary_path):
                    print(data_summary_path, 'DOES NOT EXIST!')
                    return
                df = pkl.load(open(data_summary_path, 'rb'))
                if len(model_name_filter):
                    keep_models = set(model_name_filter) | {'WF model', 'ID'}
                    df = df[df['model name'].isin(keep_models)]

                # Filter data for this subplot
                if test_generation==-1:
                    df_filtered = df[(df['result_metric_name'] == result_metric) & (df['snp type'] == snp_type)]
                    df_filtered = (
                        df_filtered
                        .groupby(['dataset_name', 'snp type', 'submodel_num','result_metric_name', 'dataset_type',
                       'p_{LD} influence', 'model name', 'Chrom'], as_index=False)
                        .agg(result_metric=("result_metric", "mean"))
                    )
                    # TODO set result_metric to mean result_metric for different df['TestGen'] in df_filtered (so that df is not affected)
                else:
                    df_filtered = df[(df['result_metric_name'] == result_metric) & (df['snp type'] == snp_type)&(df['TestGen'] == test_generation)]
                #print(df['result_metric'] )
                #print(df_filtered['submodel_num'].unique(),df['model name'].unique())
                if vae_add!='':
                    df_filtered.loc[df_filtered["model name"] == "w", "model name"] = "w" +' '+ vae_add
                    model_colors["w " + vae_add]=default_palette[2+k]
                    df_filtered = df_filtered[(df_filtered['model name'] == "w" +' '+ vae_add)]

                for key, value in rename.items():
                    # df_model['model plot name'][df_model['model plot name']==key]=value
                    df_filtered['model name'] = df_filtered['model name'].replace({key: value})

                all_frames.append(df_filtered)
            global_df = pd.concat(all_frames, ignore_index=True)
            # Plot using Seaborn's pointplot

            model_names_in_palette=model_colors.keys()
            for n_i, name in enumerate(sorted( set(global_df['model name'].unique()).difference(model_names_in_palette))):
                model_colors[name] = default_palette[2 + len(plot_paths)-1+n_i]
            #'''
            sns.pointplot(
                data=global_df,
                x='Chrom',
                y='result_metric',
                hue='model name',
                ax=ax,
                markers=['x' if model == 'WF model' else 'o' for model in global_df['model name'].unique()],
                palette=[model_colors.get(model, 'gray') for model in global_df['model name'].unique()],
                dodge=0.3,
                linestyles='',
                #scale=1.0,  # dot size
                #errwidth=1,  # thin error bars
                markersize=6,
                err_kws={'linewidth': 1},
                capsize=0.1,
            )
            for collection in ax.collections:
                collection.set_edgecolor("white")
                collection.set_linewidth(0.6)

            # Force grid lines on both x and y axes
            # ax.grid(True, which='both', linestyle='--', linewidth=0.7)
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            ax.grid(False, which='minor')
            '''
            sns.scatterplot(data=global_df,
                x='Chrom',
                y='result_metric',
                hue='model name',
                ax=ax,)
            '''
            # Force grid lines on both x and y axes

            ax.set_title(snp_types_print[j])  # {snp_type_reverse[j]}



            ax.set_xlabel("Chromosome")
            ax.set_ylabel(result_metrics_plot[i])

            #fig.text(0.08 + (i*2+j) * 0.2425, 0.86 , subplot_labels[i*2+j], fontsize=10, weight='bold') #- i * 0.45
            ax.legend().remove()
    # Get handles + labels from the last axis
    handles, labels = ax.get_legend_handles_labels()

    # Add global legend at top center
    dummy = Line2D([], [], linestyle='none')

    handles = [dummy] + handles
    labels = ["Method:"] + labels
    legend = fig.legend(
        handles,
        labels,
        # title="LD estimator",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        # bbox_to_anchor=(0.5, 1.08)
        ncol=len(labels),  # 3,  # adjust depending on how many legend entries you have

        handletextpad=0.4,
        columnspacing=1.2,
        frameon=True,
        fancybox=True
    )
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#e0e0e0')
    frame.set_linewidth(1)

    for ax in axes.flatten():
        ax.spines['top'].set_linewidth(0.4)  # .set_visible(False)
        ax.spines['right'].set_linewidth(0.4)  # .set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.tick_params(width=0.8)


    #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave some space at the top for subplot labels
    #fig.suptitle(title)
    plt.subplots_adjust(left=0.08,right=0.98,bottom=0.18,top=0.8,wspace=0.4)
    print('plot this :)')

    # Adjust layout
    #plt.tight_layout()

    fig.savefig(plot_path_out +'_fancy_gen_perf_complete.png')
    fig.savefig(plot_path_out + '_fancy_gen_perf_complete.pdf',bbox_inches='tight')
    fig.savefig(plot_path_out + '_fancy_gen_perf_complete.tiff',dpi=300,bbox_inches='tight')
    plt.show()


def load_spearman_data(plot_path,correlation_key,l_idx,min_max_best,model_name_filter=[],rename=dict(),noise_str=''):
    data_summary_path = plot_path + str(l_idx) + 'LD_spearman_corr.pkl'
    [df_model, df_afc_, df_link_] = pkl.load(open(data_summary_path, 'rb'))

    for key, value in rename.items():
        df_model['model plot name'] = df_model['model plot name'].replace({key: value})

    if len(model_name_filter):
        keep_models = set(model_name_filter)
        df_model = df_model[df_model['model plot name'].isin(keep_models)].copy()

    df_model['Spearman roh'] = df_model['Spearman roh'].abs()

    group_cols = [
        'dataset_name', 'dataset_type', 'p_{LD} influence',
        'submodel_num', 'LD estimator', 'model name','model plot name'
    ]

    df_model_best_case = df_model.groupby(
        group_cols  # ,'model plot name'
    ).agg(max_spearman=(correlation_key, min_max_best)).reset_index()
    #['Performance on all SNPs', 'Maximum performance after SNP filtering']
    df_model_best_case['Performance']='Filtered SNPs'

    #df_model_worst_case = df_model[df_model['AFC filter'] == 0].copy()
    #df_model_worst_case.rename(columns={correlation_key: 'max_spearman'}, inplace=True)
    #df_model_worst_case['Scenario']='Worst case'

    df_model_worst_case = (
        df_model[df_model['AFC filter'] == 0]
        .groupby(group_cols)
        .agg(max_spearman=(correlation_key, min_max_best))
        .reset_index()
    )

    df_model_worst_case['Performance'] = 'All SNPs'

    df_combined=pd.concat([df_model_best_case, df_model_worst_case],ignore_index=True)
    df_combined['dataset_type'] = df_combined['dataset_type'].replace('(1000,1000)', 'no noise')
    df_filter = df_combined[df_combined['dataset_type'] == noise_str]
    return df_filter

def heat_map_analysis(plot_path,correlation_key,correlation_plot_key,l_idx,min_max_best,model_name_filter=[],rename=dict(),noise_str=''):
    df= load_spearman_data(plot_path, correlation_key, l_idx, min_max_best, model_name_filter, rename,
                                     noise_str)
    split_cols = df['model plot name'].str.split('-', expand=True)


    df['dataset']="$n_{LD}="+df['dataset_name']+"$"
    df["$\\beta$"]=split_cols[1].astype(int)
    df["$M$"]= split_cols[0].astype(int)
    datasets=sorted(df['dataset'].unique(),key=lambda x:int(x.replace('$','').replace('n_{LD}=','')))

    # filter best case:
    group_cols = [
        'dataset_name', 'dataset_type', 'p_{LD} influence',
         'LD estimator', 'model plot name','dataset',"$\\beta$","$M$"
    ]

    correlation_key = 'max_spearman'
    df = (df[df['Scenario'] == 'Worst case']
        .groupby(group_cols)
        .agg(max_spearman=(correlation_key, 'mean'))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 4, figsize=(10, 4), sharex=True, sharey=True)
    # create axis for shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, (ax, ds) in enumerate(zip(axes.flatten(), datasets)):
        pivot = df[df["dataset"] == ds].pivot(
            index="$\\beta$", columns="$M$", values=correlation_key
        )
        sns.heatmap(
            pivot,
            annot=True,
            cmap="viridis",
            vmin=0.2,
            vmax=0.90,
            ax=ax,
            #cbar = (i == len(datasets) - 1)
            cbar=(i == 0),        # only first draws it
            cbar_ax=None if i else cbar_ax
        )
        ax.set_title(ds)
        ax.set_xlabel("$M$")
        if not i:
            print('huhu')
            ax.set_ylabel("$\\beta$")
        else:
            ax.set_ylabel("")

    #for ax in axes:
        ax.spines['top'].set_linewidth(0.4)  # .set_visible(False)
        ax.spines['right'].set_linewidth(0.4)  # .set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        #ax.tick_params(width=0.8)

    plt.subplots_adjust(left=0.08,right=0.9,bottom=0.18,top=0.8,wspace=0.1)

    fig.savefig(plot_path +'_heat.png')
    fig.savefig(plot_path + '_heat.pdf',bbox_inches='tight')
    fig.savefig(plot_path + '_heat.tiff',dpi=300,bbox_inches='tight')
    plt.show()

def statistic_mixed_effect_models(plot_path,correlation_key,correlation_plot_key,l_idx,min_max_best,model_name_filter=[],rename=dict(),noise_str='',add_simulation_results=False):
    by = ['M_int','beta_int']
    ascending = [True, True]
    df_combined = load_spearman_data(plot_path, correlation_key, l_idx, min_max_best, model_name_filter, rename,
                                     noise_str)
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42
    })

    subplot_labels=['A','B','C','D']


    split_cols = df_combined['model plot name'].str.split('-', expand=True)

    df_combined['M_int'] = split_cols[0].astype(int)
    df_combined['beta_int'] = split_cols[1].astype(int)
    df_combined = df_combined.sort_values(by=by,ascending=ascending)
    df_combined['x_label'] = ( "$M=" + split_cols[0] + "$\n$\\beta=" + split_cols[1] + "$")
    df_combined = df_combined.drop(columns=['M_int', 'beta_int'])

    # filter according to best case:
    df_combined = df_combined[df_combined['Scenario'] == 'Best case']
    df_combined.rename(columns={"model plot name": "model_plot_name"}, inplace=True)
    # df['dataset']="$n_{LD}="+df['dataset_name']+"$"
    # Mixed Effects Model
    model = smf.mixedlm(
        "max_spearman ~ model_plot_name",
        df_combined,
        groups=df_combined["dataset_name"]
    )
    result = model.fit()

    print(result.summary())
    ##################################################################################
    # Predicted marginal means für jede Konfiguration
    # 2. Predicted Marginal Means (EMM)
    emm = df_combined[["model_plot_name"]].drop_duplicates().copy()
    emm["predicted"] = result.predict(emm)

    # Optional: 95% CI aus Fixed Effects Standard Error
    # result.bse_fe enthält Standardfehler der Koeffizienten
    intercept = result.fe_params['Intercept']
    bse = result.bse_fe

    # Berechne EMM CI
    emm_ci = []
    for name in emm["model_plot_name"]:
        if name == df_combined["model_plot_name"].unique()[0]:
            ci_lower = intercept - 1.96*bse['Intercept']
            ci_upper = intercept + 1.96*bse['Intercept']
        else:
            coef_name = f"model_plot_name[T.{name}]"
            coef = result.fe_params.get(coef_name, 0)
            se = result.bse_fe.get(coef_name, 0)
            ci_lower = (intercept + coef) - 1.96*se
            ci_upper = (intercept + coef) + 1.96*se
        emm_ci.append((ci_lower, ci_upper))
    emm['ci_lower'] = [c[0] for c in emm_ci]
    emm['ci_upper'] = [c[1] for c in emm_ci]

    # 3. Plot
    plt.figure(figsize=(12,6))
    sns.boxplot(
        data=df_combined,
        x='model_plot_name',
        y='max_spearman',
        showfliers=False,
        palette="coolwarm"
    )

    # Overlay EMM als Punkte + Fehlerbalken
    plt.errorbar(
        x=np.arange(len(emm)),
        y=emm['predicted'],
        yerr=[emm['predicted'] - emm['ci_lower'], emm['ci_upper'] - emm['predicted']],
        fmt='o', color='black', capsize=5, label="Estimated Mean ± 95% CI"
    )

    plt.xticks(rotation=45)
    plt.ylabel("Spearman Correlation")
    plt.xlabel("Model Configuration")
    plt.title("Robust Performance Across Datasets")
    plt.legend()
    plt.tight_layout()
    plt.show()


def box_plot_for_robust_choice(plot_path,correlation_key,correlation_plot_key,l_idx,min_max_best,model_name_filter=[],rename=dict(),noise_str='',add_simulation_results=False):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    by = ['M_int','beta_int']
    ascending = [True, True]
    df_combined=load_spearman_data(plot_path,correlation_key,l_idx,min_max_best,model_name_filter,rename,noise_str)
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42
    })

    #subplot_labels=['A','B','C','D']


    split_cols = df_combined['model plot name'].str.split('-', expand=True)

    df_combined['M_int'] = split_cols[0].astype(int)
    df_combined['beta_int'] = split_cols[1].astype(int)
    df_combined = df_combined.sort_values(by=by,ascending=ascending)
    df_combined['x_label'] = ( "$M=" + split_cols[0] + "$\n$\\beta=" + split_cols[1] + "$")
    df_combined = df_combined.drop(columns=['M_int', 'beta_int'])

    if add_simulation_results:
        fig, axis = plt.subplots(1, 3, figsize=(13, 3))
        ax=axis[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    correlation_key='max_spearman'



    #print(df_combined)

    sns.boxplot(
        data=df_combined,
        x='x_label',
        y=correlation_key,
        showfliers=False,
        hue='Performance',
        ax=ax,
        dodge = 0.3,
    )
    # filter according to best case:
    #df_combined=df_combined[df_combined['Scenario'] == 'Best case']
    df_combined.rename(columns={"model plot name": "model_plot_name"}, inplace=True)
    model = smf.mixedlm(
        "max_spearman ~ model_plot_name",
        df_combined,
        groups=df_combined["dataset_name"]
    )
    result = model.fit()
    print(result.summary())

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel(correlation_plot_key)

    # Dezente Striche an jedem Tick
    ax.tick_params(
        axis='x',
        which='major',
        bottom=True,
        #direction='in',  # nach innen
        direction='out',
        length=4,  # sehr kurz
        width=0.8,  # dünn
        color='0.3'  # leicht grau statt schwarz
    )
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('0.3')


    if not add_simulation_results:

        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()
        # Add global legend at top center
        dummy = Line2D([], [], linestyle='none')

        handles = [dummy] + handles
        labels = ["Performance:"] + labels
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),  # 3,  # adjust depending on how many legend entries you have
            handletextpad=0.4,
            columnspacing=1.2,
            frameon=True,
            fancybox=True
        )
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('#e0e0e0')
        frame.set_linewidth(1)


        ax.spines['top'].set_linewidth(0.4)  # .set_visible(False)
        ax.spines['right'].set_linewidth(0.4)  # .set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        #ax.tick_params(width=0.8)

    else:
        #####################################
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()
        # Add global legend at top center
        dummy = Line2D([], [], linestyle='none')

        handles = [dummy] + handles
        labels = ["Performance of:"] + labels
        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=len(labels),  # 3,  # adjust depending on how many legend entries you have
            handletextpad=0.4,
            columnspacing=1.2,
            frameon=True,
            fancybox=True
        )
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('#e0e0e0')
        frame.set_linewidth(1)
        #ax.set_title(subplot_labels[0],loc="left",fontweight="bold")
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)

        #####################################


        # fill with simulation results:
        data_summary_path = plot_path + 'simulation_results.pkl'
        df = pkl.load(open(data_summary_path, 'rb'))
        print(df.columns)
        # delete wf model data:
        df = df[~df['model name'].isin({'WF model'})]
        df = df[~df['model name'].isin({'ID'})]
        # create x-axis name
        split_cols = df['model name'].str.split('-', expand=True)
        #df['x_label'] = ("$M=" + split_cols[0] + "$\n$\\beta=" + split_cols[1] + "$")
        df['M_int'] = split_cols[0].astype(int)
        df['beta_int'] = split_cols[1].astype(int)
        df = df.sort_values(by=by, ascending=ascending)
        df['x_label'] = ("$M=" + split_cols[0] + "$\n$\\beta=" + split_cols[1] + "$")
        df= df.drop(columns=['M_int', 'beta_int'])

        # rename:
        for old_name, new_name in zip(['selected', 'unselected'],['Targets', 'Non-targets']):
            df['snp type'] = df['snp type'].replace({old_name: new_name})


        print(df.columns)
        for i, (result_metric,y_ax_names)  in enumerate(zip(['relative_mean_error', 'relative_std_error'],['Relative distance: mean','Relative distance: standard deviation'])):
            ax=axis[1+i]
            df_filtered = df[(df['result_metric_name'] == result_metric)]
            df_filtered = (
                df_filtered
                .groupby(['dataset_name', 'snp type', 'submodel_num', 'result_metric_name', 'dataset_type',
                          'p_{LD} influence', 'model name', 'Chrom','x_label'], as_index=False,sort=False)
                .agg(result_metric=("result_metric", "mean"))
            )
            sns.boxplot(
                data=df_filtered,
                x='x_label',
                y='result_metric',
                showfliers=False,
                hue='snp type',
                ax=ax,
                dodge=0.3,
            )
            df_combined = df_filtered#[df_filtered['snp type'] == 'selected']# 'unselected'
            df_combined.rename(columns={"model name": "model_name"}, inplace=True)
            df_combined['neg_metric'] = -df_combined['result_metric']

            model = smf.mixedlm(
                "neg_metric ~ model_name",
                df_combined,
                groups=df_combined["dataset_name"]
            )
            result = model.fit()
            print(result.summary())

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.set_xlabel("Model Configuration")
            ax.set_ylabel(y_ax_names)
            #ax.legend().remove()
            #####################################
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()
            # Add global legend at top center
            dummy = Line2D([], [], linestyle='none')

            handles = [dummy] + handles
            labels = ["SNP type:"] + labels
            legend = ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.2),
                ncol=len(labels),  # 3,  # adjust depending on how many legend entries you have
                handletextpad=0.4,
                columnspacing=1.2,
                frameon=True,
                fancybox=True
            )
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('#e0e0e0')
            frame.set_linewidth(1)
            #####################################
            #ax.set_title(subplot_labels[1 + i],loc="left",fontweight="bold")
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            # Dezente Striche an jedem Tick
            ax.tick_params(
                axis='x',
                which='major',
                bottom=True,
                # direction='in',  # nach innen
                direction='out',
                length=4,  # sehr kurz
                width=0.8,  # dünn
                color='0.3'  # leicht grau statt schwarz
            )
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.spines['bottom'].set_color('0.3')


        plt.subplots_adjust(top=0.75)

    plt.subplots_adjust(left=0.08,right=0.98,bottom=0.18,top=0.8,wspace=0.2)

    fig.savefig(plot_path +'_hyper_box.png')
    fig.savefig(plot_path + '_hyper_box.pdf',bbox_inches='tight')
    fig.savefig(plot_path + '_hyper_box.tiff',dpi=300,bbox_inches='tight')
    plt.show()
    print(plot_path + '_hyper_box.pdf')





