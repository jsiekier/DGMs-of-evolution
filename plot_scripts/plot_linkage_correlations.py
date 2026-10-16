import os
import seaborn as sns
from nice_figures import plot_experimental_max_corr, \
    plot_synthetic_DS_performance, plot_synthetic_max_corr, plot_fancy_performance_experimental



sns.set_style("whitegrid")




if __name__ == '__main__':
    columns_model = ['dataset_name', 'dataset_type','p_{LD} influence','model name', 'submodel_num','AFC filter','Spearman roh','Chrom']
    columns_afc = ['dataset_name', 'dataset_type','p_{LD} influence','AFC filter','Spearman roh']

    show_random=False
    model_name_filter=[]

    rename = {'w':'Focal+Context','no w':'Focal'}
    #plot_path=[(
    synthetic = 1
    plot_path=  ['../plots/13exp_testset_posneg_mv/','../plots/synthetic_new_models_posneg_filter_new/'][synthetic]
    use_old=False

    if synthetic:
        pass
        plot_synthetic_DS_performance(plot_path,use_old=use_old,rename=rename)
    else:
        plot_path = [(plot_path, '')]

        result_metrics_plot=[['Relative distance: mean','Relative distance: standard deviation']]#],['mean','standard deviation'],['MAE','RMSE']
        for result_metrics,title_met,y_ax_names in zip([['relative_mean_error', 'relative_std_error'],['mean','std'],['MAE','RMSE']],['rel','distr','mae'],result_metrics_plot):
            for test_gen in [-1]:#,0,1]:
                if test_gen>=0:
                    title='Test gen:'+str(test_gen)
                else:
                    title=''
                plot_path_tmp=plot_path[0][0]+title_met+'_'+str(test_gen)

                plot_fancy_performance_experimental(plot_path,plot_path_tmp,result_metrics=result_metrics,
                                              test_generation=test_gen,
                                              title=title,
                                              result_metrics_plot=y_ax_names,model_name_filter=model_name_filter,rename=rename)


    rename = {'w': 'VAE','random':''}
    res_metrics=[('MAE','MAE','min'),
                 ('Spearman roh',r"Spearman $\rho$",'max'),
                 ('Spearman roh diffcomp',r"Weighted mean Spearman $\bar{\rho}$",'max'),
                 ('XI', r'$\xi$', 'max')]

    filter_modes = [ 'coherence','afc_geom','LDx_mask','afc_min','afc_max','afc_sf_min']# , #,
    show_modes=['r^2','r','r2_r']
    promising_combis=[(1, 3, 1) ]

    input_paths = (
        [(os.fspath(plot_path), '')]
        if isinstance(plot_path, (str, os.PathLike))
        else list(plot_path)
    )
    filter_model_names=['VAE+LDx']
    for combi in promising_combis:
        show_mode,filter_mode,(correlation_key,correlation_plot_key,min_max_best)=show_modes[combi[2]],filter_modes[combi[1]],res_metrics[combi[0]]
        print(filter_mode,correlation_plot_key,show_mode)
        plot_path_ = input_paths[0][0] #os.path.join(input_paths[0][0], 'gt_')
        print(plot_path_)
        if synthetic:
            plot_path = [(plot_path, '')]
            for noise_str,roh in zip(['no noise','(100,40)'],[0.75,0.55]):#'(200,50)',
                plot_synthetic_max_corr(input_paths, correlation_key, correlation_plot_key, min_max_best,
                                           model_name_filter=model_name_filter, rename=rename, plot_path=plot_path_,
                                        noise_str=noise_str,filter_mode=filter_mode,show_mode=show_mode,
                                        filter_model_names=filter_model_names,add_quality_coverage_panels=3,rho_target=roh)
        else:

            plot_experimental_max_corr(plot_path, correlation_key,correlation_plot_key,min_max_best,
                                       model_name_filter=model_name_filter,rename=rename,plot_path=plot_path,
                                       filter_mode=filter_mode,show_mode=show_mode,filter_model_names=filter_model_names,
                                       add_quality_coverage_panels=3,rho_target=0.75)



