import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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








def plot_dataset_details_box(datas, data_subtitles, x_axis_names, plt_out_name):
    locations=[]
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
            if i==0 and j==0:
                plot_df=plot_df[plot_df['Type']=='$\mathcal{N}_{50}$']
            sns.boxplot(data=plot_df, x="Dataset configuration", y=sup_title, hue="Type", ax=axis[i, j], gap=.1,showfliers=False)
            #sns.violinplot(data=plot_df, x='', y=sup_title, ax=axis[i, j], inner="quartile",hue='Type', density_norm="width",split=True)

            #axis[i, j].set_title(sup_title, fontsize=10)
            axis[i, j].tick_params(axis='x', rotation=45)
            if j==1:#not (i==1 and j==0):
                axis[i, j].legend(loc='upper right')  # Set legend location
            if i == 0 and j == 0:
                axis[i,j].legend().set_visible(False)
            # Add subplot label
            #fig.text(0.06 + j * 0.498, 0.94 - i * 0.46, subplot_labels[data_idx], fontsize=12, weight='bold')
            fig.text(0.06 + j * 0.498, 0.94 - i * 0.41, subplot_labels[data_idx], fontsize=12, weight='bold')

    # Adjust layout to ensure labels fit nicely
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave some space at the top for subplot labels
    #plt.show()

    #if plt_out_name:
    plt.savefig(plt_out_name+'.png')
    plt.savefig(plt_out_name+'.pdf')
    plt.savefig(plt_out_name+'.tiff')
    print(plt_out_name+'.png')