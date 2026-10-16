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

custom_palette = {
    'Baseline': "#4C72B0",  # Blue
    'VAE': "#DD8452",  # Orange
    'LDx': "#C44E52",  # Red
    'random': "#55A868",  # Green
}
sns.set_style("white")





if __name__ == '__main__':
    res_metrics=[('MAE','MAE','min'),
                 ('Spearman roh',r"Spearman $\rho$",'max'),
                 ('Spearman roh diffcomp',r"Weighted mean Spearman $\bar{\rho}$",'max'),
                 ('XI', r'$\xi$', 'max')]
    filter_modes = [ 'coherence','afc_geom','LDx_mask','afc_min','afc_max','afc_sf_min']# , #,
    #show_modes=['r^2','r','r2_r']
    filter_model_names = ['VAE+LDx']
    promising_combis = [(1,3),(2, 0)]
    combi=promising_combis[0]
    filter_mode, (correlation_key, correlation_plot_key, min_max_best) = filter_modes[combi[1]], res_metrics[combi[0]]


    synthetic = 0

    plot_path=   ['../plots/13exp_testset_posneg_mv_zero/','../plots/synthetic_new_models_posneg_filter_new/']#[synthetic]
    #['../plots/13exp_testset_posneg_filter_koh_mean_LDx_r/', '../plots/synthetic_new_models_posneg_filter_koh_mean_ldx/']#

    #'../plots/exp_new_models_other_testset/'
    #../plots/synthetic_new_models_gt_plus_noise/
    source = 'LD estimator'
    _, df_afc_ldx, _ = pkl.load(open(f'{plot_path[synthetic]}3LD_spearman_corr.pkl', 'rb'))
    df_model, df_afc_baseline, _ = pkl.load(open(f'{plot_path[synthetic]}3LD_spearman_corr.pkl', 'rb'))

    df_model = df_model.copy()
    if 'filter' in df_model.columns:
        df_model = df_model[(df_model['filter'] == filter_mode)]
        df_afc_baseline = df_afc_baseline.loc[(df_afc_baseline[source] != 'org_sim') | (df_afc_baseline['filter'] == filter_mode)].copy()
    mask = df_model[source] != 'VAE+LDx'
    df_model.loc[mask, source] = df_model.loc[mask, 'model plot name']


    df_model[correlation_key] = df_model[correlation_key].abs()

    for df in (df_afc_ldx, df_afc_baseline):
        df['submodel_num'] = 0
        df['model name'] = ''
        df[correlation_key] = df[correlation_key].abs()

    df_afc_ldx = df_afc_ldx[df_afc_ldx[source] == 'LDx']
    df_afc_baseline = df_afc_baseline[df_afc_baseline[source] == 'org_sim']

    df_combined = pd.concat([df_model, df_afc_ldx, df_afc_baseline], ignore_index=True)
    df_combined[source] = df_combined[source].replace({'org_sim': 'Baseline'})
    df_combined['num_eval_spns'] = pd.to_numeric(df_combined['num_eval_spns'], errors='coerce')
    df_combined[correlation_key] = pd.to_numeric(df_combined[correlation_key], errors='coerce')
    df_combined = df_combined[df_combined['Chrom'] == '2L']
    for filter_model_name in filter_model_names:
        df_combined = df_combined[df_combined[source] != filter_model_name]


    fig, axis= plt.subplots(1, 2, figsize=(9.5, 2.5),sharey=True)
    axes=axis[1]
    df_combined = df_combined.copy()
    #vae = df_combined["LD estimator"].eq("VAE")
    #df_combined.loc[vae, "num_eval_spns"] = (
    #    df_combined.loc[vae]
    #    .groupby("AFC filter")["num_eval_spns"]
    #    .transform("median")
    #    .astype(int)
    #    .to_numpy()
    #)
    sns.lineplot(data=df_combined, x='num_eval_spns', y=correlation_key, hue=source, ax=axes, palette=custom_palette)
    axes.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    axes.grid(False, which='minor')

    axes.set_xlabel("Number of evaluated pairs on 2L of empirical DS")
    axes.set_ylabel("")
    axes.legend().remove()
    #########################################################################################################################
    synthetic = 1

    _, df_afc_ldx, _ = pkl.load(open(f'{plot_path[synthetic]}3LD_spearman_corr.pkl', 'rb'))
    df_model, df_afc_baseline, _ = pkl.load(open(f'{plot_path[synthetic]}3LD_spearman_corr.pkl', 'rb'))

    df_model = df_model.copy()
    df_model[source] = df_model['model plot name']
    df_model[correlation_key] = df_model[correlation_key].abs()

    for df in (df_afc_ldx, df_afc_baseline):
        df['submodel_num'] = 0
        df['model name'] = ''
        df[correlation_key] = df[correlation_key].abs()

    df_afc_ldx = df_afc_ldx[df_afc_ldx[source] == 'LDx']
    df_afc_baseline = df_afc_baseline[df_afc_baseline[source] == 'org_sim']

    if 'filter' in df_model.columns:
        df_model = df_model[(df_model['filter'] == filter_mode)]
        df_afc_baseline = df_afc_baseline.loc[(df_afc_baseline[source] != 'org_sim') | (df_afc_baseline['filter'] == filter_mode)].copy()

    df_combined = pd.concat([df_afc_ldx, df_afc_baseline,df_model], ignore_index=True)#
    for filter_model_name in filter_model_names:
        df_combined = df_combined[df_combined[source] != filter_model_name]
    df_combined[source] = df_combined[source].replace({'org_sim': 'Baseline','w': 'VAE'})
    df_combined['num_eval_spns'] = pd.to_numeric(df_combined['num_eval_spns'], errors='coerce')

    df_combined[correlation_key] = pd.to_numeric(df_combined[correlation_key], errors='coerce')
    print(df_combined.head())
    df_combined = df_combined[df_combined['dataset_type'] == '(1000,1000)']
    #df_combined = df_combined[df_combined['dataset_type'] == '(100,40)']
    df_combined = df_combined[df_combined['dataset_name'] == '8']
    df_combined = df_combined[df_combined['p_{LD} influence'] == '25']




    axes=axis[0]
    #df_combined = df_combined.copy()
    #vae = df_combined["LD estimator"].eq("VAE")
    #df_combined.loc[vae, "num_eval_spns"] = (
    #    df_combined.loc[vae]
    #    .groupby("AFC filter")["num_eval_spns"]
    #    .transform("median")
    #    .astype(int)
    #    .to_numpy()
    #)
    v = df_combined.copy()
    #v = df_combined[df_combined[source].eq("VAE")]
    v = v[v["train pairs"].isna()].copy()

    print(
        v.groupby(["AFC filter", "num_eval_spns"], dropna=False)
        .agg(
            n=(correlation_key, "size"),
            rho_min=(correlation_key, "min"),
            rho_max=(correlation_key, "max")
        )
        .reset_index()
        .sort_values(["AFC filter", "num_eval_spns"])
        .to_string(index=False)
    )
    sns.lineplot(data=v, x='num_eval_spns', y=correlation_key, hue=source, ax=axes, palette=custom_palette)
    axes.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    axes.grid(False, which='minor')

    axes.set_xlabel("Number of evaluated pairs of synthetic DS ($n_{LD}=.08,|T|=25$)")
    #axes.set_ylabel("Spearman's $\\rho$")
    axes.set_ylabel(correlation_plot_key)


    axes.legend().remove()
    # Get handles + labels from the last axis
    handles, labels = axes.get_legend_handles_labels()

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

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.88, wspace=0.1)#
    #plt.tight_layout()
    plt.savefig('LD_filtering_debug_gt_new.pdf')
    plt.show()
