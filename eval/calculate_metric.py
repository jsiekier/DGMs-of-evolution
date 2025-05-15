import numpy as np

from numpy import inf

def calculate_metric(raw_linkage_metric,use_distance=False):
    # read most important data:
    (eval_positions,all_n_f,all_n_t,all_n_t_distance,all_n_f_distance,all_n_t_effect,afc,all_f_t,all_f_t_distance,all_n_afc,all_n_f_rsquare,all_n_f_D_normal)=raw_linkage_metric
    alphas=[1]#[1,2,3,4,5,6,10,20,50]
    # naive def 1 without distance_data:
    # step 1 summ targets:
    max_distance_n_t=0
    for row in all_n_t_distance:
        for e in row:
            max_dis=max(e)
            if max_dis>max_distance_n_t:
                 max_distance_n_t=max_dis
    max_distance_n_f=np.max(np.asarray(all_n_f_distance))


    for alpha in alphas:
        linkage_def1=[]
        linkage_def_kramer=[]
        linkage_to_target=[]




        for n_f,n_t,n_t_distance,n_f_distance,n_t_effect in zip(all_n_f,all_n_t,all_n_t_distance,all_n_f_distance,all_n_t_effect):
            #n_t_mean=np.asarray([ sum(entry)/len(entry) for entry in n_t ])
            n_t_mean=[]
            for n_t_entry, n_t_distance_entry,n_t_effect_entry in zip(n_t,n_t_distance,n_t_effect):
                weighted_linkage=0
                for e, d, effect in zip(n_t_entry, n_t_distance_entry,n_t_effect_entry):
                    if not d:
                        weighted_linkage+=e*effect
                    else:
                        if use_distance:
                            weighted_linkage+=e*effect*(1-(np.log(d)/np.log(max_distance_n_t)))
                        else:
                            weighted_linkage += e * effect
                n_t_mean.append(weighted_linkage)
            n_t_mean=np.asarray(n_t_mean)


            metric=n_f*n_t_mean#
            if use_distance:
                metric = np.sum(metric * (1-(np.log(n_f_distance) /np.log( max_distance_n_f))))
            else:
            #metric=np.sum(metric * (1-(n_f_distance /max_distance_n_f)))
                metric=np.mean(metric)
            linkage_def1.append(metric)


            metric=n_f+alpha*n_f*n_t_mean
            if use_distance:
                metric = np.sum(metric * (1-(np.log(n_f_distance) /np.log( max_distance_n_f))))
            else:
                metric=np.mean(metric)
            linkage_def_kramer.append(metric)

        all_f_t =np.asarray(all_f_t)
        all_f_t_distance = np.asarray(all_f_t_distance)
        f_t_distance_max = np.max(all_f_t_distance)


        all_f_t_distance= np.log(all_f_t_distance)/np.log(f_t_distance_max)
        all_f_t_distance[all_f_t_distance == -inf] = 0
        all_n_t_effect=np.asarray(all_n_t_effect)
        #n_t_effect=np.asarray(n_t_effect)
        print('all_n_t_effect.shape',all_n_t_effect.shape)
        effect= np.expand_dims(np.asarray(all_n_t_effect)[0,0],axis=0)
        #effect = np.expand_dims(np.asarray(n_t_effect)[0], axis=0)

        print(all_f_t_distance.shape, all_f_t.shape,effect.shape )

        if use_distance:
            no_window_linkage_def=np.sum(all_f_t*(1-all_f_t_distance)*effect,axis=-1)
        else:
            no_window_linkage_def = np.mean(all_f_t  * effect, axis=-1)
        tmp_distance=(linkage_def1/np.max(linkage_def1))-(no_window_linkage_def/np.max(no_window_linkage_def))

        #plot_correlation(afc,tmp_distance,'alpha:'+str(alpha))

        all_n_afc=np.asarray(all_n_afc)
        all_n_f=np.asarray(all_n_f)

        afc_neighbor_metric=np.mean(np.abs(all_n_afc),axis=-1)

        afc_neighbor_linkage_corr=np.mean(np.abs(all_n_afc)*all_n_f,axis=-1)

        #save linkage metric:
        result=[eval_positions,afc,linkage_def1,linkage_def_kramer,
                no_window_linkage_def,tmp_distance,afc_neighbor_metric,afc_neighbor_linkage_corr,all_n_f,all_n_f_rsquare,all_n_f_D_normal]
        return result







