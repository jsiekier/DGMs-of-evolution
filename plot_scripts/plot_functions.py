import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.colors as mcolors

'''
matplotlib.use('TkAgg')

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
sns.set_style("darkgrid", {'axes.grid': True})
'''






# Define your custom colors
seaborn_orange = sns.color_palette("deep")[1]  # seaborn's default orange
seaborn_blue = sns.color_palette("deep")[0]   # seaborn's default blue

# Create faded versions by adjusting the alpha (blending with white)
def fade_color(color, alpha=0.5):
    r, g, b = color
    return (1 - alpha) * 1.0 + alpha * r, (1 - alpha) * 1.0 + alpha * g, (1 - alpha) * 1.0 + alpha * b

seaborn_orange_fade = fade_color(seaborn_orange, alpha=0.5)
seaborn_blue_fade = fade_color(seaborn_blue, alpha=0.5)

# Combine in your custom palette
custom_palette = [seaborn_orange, seaborn_blue, seaborn_orange_fade, seaborn_blue_fade]


def plot_dataset_details_box(datas, data_subtitles, x_axis_names, plt_out_name,plt_type='violin'):
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



    locations=[]
    #fig, axis = plt.subplots(2, 2, figsize=(15, 8),sharex=True)
    fig, axis = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
    subplot_labels = ['A', 'B', 'C', 'D']

    for data_idx, (data, sup_title_other) in enumerate(zip(datas, data_subtitles)):
        i = data_idx // 2
        j = data_idx % 2
        ax=axis[i,j]

        sup_title,other,plt_type=sup_title_other

        if len(data):  # Check if there's data to plot
            # Flatten the data into a DataFrame for seaborn compatibility

            plot_data = []
            idx=0
            for sublist_idx,sublist in enumerate(data):
                for s_idx,value in enumerate(sublist):

                    arr=np.abs(np.asarray(value))
                    arr[np.isnan(arr)] = 0

                    for v in arr:
                        plot_data.append({'Dataset configuration': x_axis_names[sublist_idx], sup_title: v,'Type':other[s_idx%len(other)]})
                    idx+=1
                    #print( x_axis_name[sublist_idx])
                    #print(value)

            plot_df = pd.DataFrame(plot_data)
            #df_exploded = plot_df.explode('Value', ignore_index=True)
            print(sup_title)
            #print(plot_df)

            # Create the violin plot
            if sup_title=='LD $r^2$':
                plot_df=plot_df[plot_df['Type']=='$\mathcal{N}_{50}$']
            #
            if plt_type=='violin':
                sns.violinplot(data=plot_df, x="Dataset configuration", y=sup_title, ax=ax, inner="quartile",hue='Type', density_norm="width",split=True,cut=0)
            else:
                sns.boxplot(data=plot_df, x="Dataset configuration", y=sup_title, hue="Type", ax=ax, gap=.1,
                            showfliers=False)
            #axis[i, j].set_title(sup_title, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            if sup_title=='Estimated $|s|$' or sup_title =='SF $f^{(0)}$':#not (i==1 and j==0):
                ax.legend(loc='upper right')  # Set legend location
            if  sup_title=='LD $r^2$':
                ax.legend().set_visible(False)
            # Add subplot label
            #fig.text(0.06 + j * 0.498, 0.94 - i * 0.46, subplot_labels[data_idx], fontsize=12, weight='bold')
            #fig.text(0.06 + j * 0.498, 0.94 - i * 0.41, subplot_labels[data_idx], fontsize=12, weight='bold')
            #ax.set_title(subplot_labels[data_idx],fontsize=10, loc='left')# fontsize=12, weight='bold',
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
            ax.grid(False, which='minor')
            ax.spines['top'].set_linewidth(0.4)  # .set_visible(False)
            ax.spines['right'].set_linewidth(0.4)  # .set_visible(False)
            ax.spines['left'].set_linewidth(0.4)
            ax.spines['bottom'].set_linewidth(0.4)

    # Adjust layout to ensure labels fit nicely

    plt.tight_layout()#rect=[0, 0, 0.95, 0]
    if plt_out_name:
        plt.savefig(plt_out_name+'.png')
        plt.savefig(plt_out_name+'.pdf')
        plt.savefig(plt_out_name+'.tiff')
    print(plt_out_name+'.png')
 #rect=[0, 0, 1, 0.95] # Leave some space at the top for subplot labels
    plt.show()
def color_from_value(val, vmin, vmax, cmap_name="viridis"):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap_name)
    return cmap(norm(val))


def plot_dataset_details_box_chroms(datas, data_subtitles, plt_out_name):

    fig, axis = plt.subplots(2, 2, figsize=(15, 8),sharex=True)
    subplot_labels = ['A', 'B', 'C', 'D']

    for data_idx, (data, sup_title_other) in enumerate(zip(datas, data_subtitles)):
        i = data_idx // 2
        j = data_idx % 2

        sup_title,other=sup_title_other

        if len(data):  # Check if there's data to plot
            # Flatten the data into a DataFrame for seaborn compatibility

            plot_data = []
            idx=0
            for key,sublist in data.items():
                if len(sublist)==2 and len(other)==2:
                    for s_idx,value in enumerate(sublist):

                        arr=np.abs(np.asarray(value))
                        arr[np.isnan(arr)] = 0

                        for v in arr:
                            plot_data.append({'Chromosome': key, sup_title: v,'Type':other[s_idx%len(other)]})
                        idx+=1
                elif  len(sublist)==2:
                    value=sublist[1]
                    arr = np.abs(np.asarray(value))
                    arr[np.isnan(arr)] = 0

                    for v in arr:
                        plot_data.append({'Chromosome': key, sup_title: v, 'Type': other[0]})
                else:

                    for v in sublist:
                        plot_data.append({'Chromosome': key, sup_title: v, 'Type': other[0]})


            plot_df = pd.DataFrame(plot_data)
            print(sup_title)

            # Create the violin plot
            #if i==0 and j==0:
            #    plot_df=plot_df[plot_df['Type']=='$\mathcal{N}_{50}$']
            sns.boxplot(data=plot_df, x="Chromosome", y=sup_title, hue="Type", ax=axis[i, j], gap=.1,showfliers=False)

            axis[i, j].tick_params(axis='x', rotation=45)
            if j==1:#not (i==1 and j==0):
                axis[i, j].legend(loc='upper right')  # Set legend location
            if i == 0 and j == 0:
                axis[i,j].legend().set_visible(False)
            # Add subplot label
            fig.text(0.06 + j * 0.498, 0.94 - i * 0.41, subplot_labels[data_idx], fontsize=12, weight='bold')

    # Adjust layout to ensure labels fit nicely
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave some space at the top for subplot labels
    plt.show()

    #if plt_out_name:
    plt.savefig(plt_out_name+'.png')
    plt.savefig(plt_out_name+'.pdf')
    plt.savefig(plt_out_name+'.tiff')
    print(plt_out_name+'.png')