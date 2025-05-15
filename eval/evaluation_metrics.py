import numpy as np
from scipy.stats import wasserstein_distance,kstest

def get_density_on_batch(data, bandwidth, grid):
    data= np.repeat(data[:, :, np.newaxis], grid.shape[-1], axis=2)
    u=(grid-data)/bandwidth
    kernel_vals=np.exp(-0.5 * u ** 2)/ np.sqrt(2 * np.pi)
    density_estimates = np.sum(kernel_vals, axis=1) / (data.shape[1] * bandwidth)
    return density_estimates

def KDE_grid_distance(gt_data, prediction, gt_bandwidth=0.1, prediction_bandwidth=0.05, grid=[]):
    if not len(grid):
        grid = np.linspace(0, 1, 1000)

    grid=np.expand_dims(grid, axis=(0,1))

    #gt_bandwidth_scott = 10**(-1./5)
    #prediction_bandwidth_scott = 100**(-1./5)

    gt_bandwidth_silverman = (10 * 3 / 4.)**(-1. / 5)
    prediction_bandwidth_silverman = (100 * 3 / 4.)**(-1. / 5)

    denisty_gt = get_density_on_batch(gt_data, gt_bandwidth_silverman, grid)#gt_bandwidth
    denisty_pred = get_density_on_batch(prediction, prediction_bandwidth_silverman, grid)#prediction_bandwidth
    KDE_distance = np.mean(np.abs(denisty_gt - denisty_pred), axis=-1)
    return KDE_distance

def ks_norm_distance(gt_sample, pred_sample):
    p_val_gt=kstest(gt_sample,'norm')[1]
    p_val_pred = kstest(pred_sample, 'norm')[1]
    p_val_distance=np.abs(p_val_gt-p_val_pred)
    return p_val_distance

######################################################################################################################################
def relative_error(gt_freqs,prediction,naive_noise_id):
    # apply metric to all generations
    # shape assumption (#SNPs,#replicates)

    # increase replicate number of gt and noise id
    gt_freqs= np.tile(np.expand_dims(gt_freqs,axis=1) ,[1,10,1,])
    gt_freqs= np.reshape(gt_freqs,(gt_freqs.shape[0],gt_freqs.shape[1]*gt_freqs.shape[2]))

    naive_noise_id= np.tile(np.expand_dims(naive_noise_id,axis=1) ,[1,10,1,])
    naive_noise_id= np.reshape(naive_noise_id,(naive_noise_id.shape[0],naive_noise_id.shape[1]*naive_noise_id.shape[2]))


    error_distance1=np.abs(prediction-gt_freqs)
    error_distance2 = np.abs(gt_freqs - naive_noise_id)
    result=(error_distance1+0.0001)/(error_distance2+0.0001)
    result=np.mean(result,axis=-1)



    return result

def relative_error_mean(gt_freqs,prediction,naive_noise_id,summary_method,return_mean=True):
    # apply metric to all generations
    # shape assumption (#SNPs,#replicates)
    #print('t',gt_freqs[0][0])
    gt_freqs= np.mean(gt_freqs,axis=-1)
    naive_noise_id= np.mean(naive_noise_id,axis=-1)
    prediction = np.mean(prediction, axis=-1)

    if summary_method==0:
        error_distance1=np.abs(prediction-gt_freqs)
        error_distance2 =np.abs(gt_freqs - naive_noise_id)
        result = (error_distance1 + 0.0001) / (error_distance2 + 0.0001)
        result=np.mean(result,axis=0)
    elif summary_method==1:
        error_distance1=np.abs(prediction-gt_freqs)
        error_distance2 =np.abs(gt_freqs - naive_noise_id)
        result = error_distance1 -error_distance2

        #print('gt',gt_freqs)
        #print('noise',naive_noise_id)
        #print('pred',prediction)
        #print('res',result)
        #print('r mean',np.mean(result, axis=0))

        # ((error_distance1 + 0.0001)**(1/4)) / ((error_distance2 + 0.0001)**(1/4))#(np.log(error_distance1+ 0.0001) ) / (np.log(error_distance2 + 0.0001))
        if return_mean:
            result = np.mean(result, axis=0)#iqr(result, axis=0)#np.median(result, axis=0)


    elif summary_method==2:
        error_distance1=np.mean(np.abs(prediction-gt_freqs),axis=0)
        error_distance2 = np.mean(np.abs(gt_freqs - naive_noise_id),axis=0)
        result=(error_distance1+0.0001)/(error_distance2+0.0001)
    #print(result,type(result),flush=True)
    return result


def relative_error_std(gt_freqs,prediction,naive_noise_id,summary_method,return_mean=True):
    # apply metric to all generations
    # shape assumption (#SNPs,#replicates)

    gt_freqs= np.std(gt_freqs,axis=-1)
    naive_noise_id= np.std(naive_noise_id,axis=-1)
    prediction = np.std(prediction, axis=-1)
    if summary_method == 0:
        error_distance1=np.abs(prediction-gt_freqs)
        error_distance2 = np.abs(gt_freqs - naive_noise_id)
        result = (error_distance1 + 0.0001) / (error_distance2 + 0.0001)
        result=np.mean(result,axis=0)
    elif summary_method==1:
        error_distance1=np.abs(prediction-gt_freqs)
        error_distance2 = np.abs(gt_freqs - naive_noise_id)
        result = error_distance1 -error_distance2# ((error_distance1 + 0.0001)**(1/4)) / ((error_distance2 + 0.0001)**(1/4))#(np.log(error_distance1+ 0.0001) ) / (np.log(error_distance2 + 0.0001))
        if return_mean:
            result = np.mean(result, axis=0)#iqr(result, axis=0)#np.median(result, axis=0)
    elif summary_method==2:
        error_distance1=np.mean(np.abs(prediction-gt_freqs),axis=0)
        error_distance2 = np.mean(np.abs(gt_freqs - naive_noise_id),axis=0)
        result=(error_distance1+0.0001)/(error_distance2+0.0001)
    return result

def relative_error_wasserstein(gt_freqs,prediction,naive_noise_id,summary_method):
    # apply metric to all generations
    # shape assumption (#SNPs,#replicates)
    if summary_method == 0:
        result=[]
        for snp_num in range(gt_freqs.shape[0]):
            error_distance1=wasserstein_distance(gt_freqs[snp_num],prediction[snp_num])
            error_distance2 = wasserstein_distance(gt_freqs[snp_num],naive_noise_id[snp_num])
            res=(error_distance1+0.0001)/(error_distance2+0.0001)
            result.append(res)
        result=np.asarray(result)
        result = np.mean(result, axis=0)
    elif summary_method==1:
        result=[]
        for snp_num in range(gt_freqs.shape[0]):
            error_distance1=wasserstein_distance(gt_freqs[snp_num],prediction[snp_num])
            error_distance2 = wasserstein_distance(gt_freqs[snp_num],naive_noise_id[snp_num])
            result = error_distance1 - error_distance2  # ((error_distance1 + 0.0001)**(1/4)) / ((error_distance2 + 0.0001)**(1/4))#(np.log(error_distance1+ 0.0001) ) / (np.log(error_distance2 + 0.0001))

        result=np.asarray(result)
        result = np.mean(result,axis=0)  # iqr(result, axis=0)#np.median(result, axis=0)            result.append(res)
    elif summary_method==2:
        d1=[]
        d2=[]
        for snp_num in range(gt_freqs.shape[0]):
            error_distance1=wasserstein_distance(gt_freqs[snp_num],prediction[snp_num])
            error_distance2 = wasserstein_distance(gt_freqs[snp_num],naive_noise_id[snp_num])
            d1.append(error_distance1)
            d2.append(error_distance2)
        d1=np.mean(np.asarray(d1),axis=0)
        d2 = np.mean(np.asarray(d2), axis=0)
        result = (d1 + 0.0001) / (d2 + 0.0001)
    return result

def relative_error_ks(gt_freqs,prediction,naive_noise_id,summary_method):
    # apply metric to all generations
    # shape assumption (#SNPs,#replicates)

    d1=[]
    d2=[]
    for snp_num in range(gt_freqs.shape[0]):
        error_distance1=kstest(gt_freqs[snp_num],prediction[snp_num])[1]
        error_distance2 =kstest(gt_freqs[snp_num],naive_noise_id[snp_num])[1]
        d1.append(error_distance1)
        d2.append(error_distance2)
    d1=np.mean(np.asarray(d1),axis=0)
    d2 = np.mean(np.asarray(d2), axis=0)
    result = (d1 + 0.0001) / (d2 + 0.0001)


    return result
def relative_error_kde(gt_freqs,prediction,naive_noise_id,summary_method):
    # apply metric to all generations
    # shape assumption (#SNPs,#replicates)
    if summary_method == 0:
        error_distance1=KDE_grid_distance(gt_freqs,prediction)
        error_distance2 = KDE_grid_distance(gt_freqs,naive_noise_id)
        result = (error_distance1 + 0.0001) / (error_distance2 + 0.0001)
        result = np.mean(result, axis=0)
    elif summary_method == 1:
        error_distance1=KDE_grid_distance(gt_freqs,prediction)
        error_distance2 = KDE_grid_distance(gt_freqs,naive_noise_id)
        result = error_distance1 -error_distance2#((error_distance1+ 0.0001)**(1/4) ) / ((error_distance2+ 0.0001)**(1/4) )
        result = np.mean(result, axis=0)#iqr(result, axis=0)#np.median(result, axis=0)
    elif summary_method==2:
        error_distance1=np.mean(KDE_grid_distance(gt_freqs,prediction),axis=0)
        error_distance2 = np.mean(KDE_grid_distance(gt_freqs,naive_noise_id),axis=0)
        result=(error_distance1+0.0001)/(error_distance2+0.0001)
    return result


def kstest_pvalue(gt_sample, pred_sample):
    return kstest(gt_sample,pred_sample)[1]

class eval_metric():
    def __init__(self,internal_name,plot_name,eval_function,axis,data_type,y_axis_name,args=None):
        self.internal_name=internal_name
        self.plot_name=plot_name
        self.eval_function=eval_function
        self.axis=axis
        self.data_type=data_type
        self.args=args
        self.y_axis_name=y_axis_name

    def call(self,distributions):
        return self.eval_function(*distributions)


kde = eval_metric(internal_name='KDE',plot_name='Kernel Density Estimation',
                  eval_function=KDE_grid_distance,axis=2,data_type='',
                  args={'gt_bandwidth':0.05, 'prediction_bandwidth':0.01},
                  y_axis_name='Distance')

eml = eval_metric(internal_name='EMD',plot_name='Wasserstein Distance',
                  eval_function=wasserstein_distance,axis=1,data_type=''
                  ,y_axis_name='Distance')
ks_test = eval_metric(internal_name='KS',plot_name='KS Test',
                      eval_function=kstest_pvalue,axis=1,data_type='',y_axis_name='p-value')

ks_norm= eval_metric(internal_name='KS_norm',plot_name='KS Test with \mathcal{N}(0,1)',eval_function=ks_norm_distance,axis=1,data_type='',y_axis_name='Distance')
#################################################################################################################################
relative_naive_error=eval_metric(internal_name='relative_error',
                                 plot_name='Relative Error of reps',
                                 eval_function=relative_error,
                                 axis=3,data_type='',y_axis_name='Relative Error Replicates')

relative_naive_error_mean=eval_metric(internal_name='relative_mean_error',
                                 plot_name='Relative Error of Mean',
                                 eval_function=relative_error_mean,
                                 axis=3,data_type='',y_axis_name='Relative Error Mean')

relative_naive_error_std=eval_metric(internal_name='relative_std_error',
                                 plot_name='Relative Error of Std',
                                 eval_function=relative_error_std,
                                 axis=3,data_type='',y_axis_name='Relative Error Std')

relative_naive_error_kde=eval_metric(internal_name='relative_kde_error',
                                 plot_name='Relative Error of KDE',
                                 eval_function=relative_error_kde,
                                 axis=3,data_type='',y_axis_name='Relative Error KDE')
relative_naive_error_eml=eval_metric(internal_name='relative_emd_error',
                                 plot_name='Relative Error of EMD',
                                 eval_function=relative_error_wasserstein,
                                 axis=3,data_type='',y_axis_name='Relative Error EMD')
relative_naive_error_kstest=eval_metric(internal_name='relative_ks_error',
                                 plot_name='Relative Error of KS-test',
                                 eval_function=relative_error_ks,
                                 axis=3,data_type='',y_axis_name='Relative Error KS-test')

all_metrics={3:[relative_naive_error_mean,relative_naive_error_std],#,
               # relative_naive_error_kde],
             2:[],#[kde],
             1:[],#[eml,ks_test]
             }
all_metrics_list=[relative_naive_error_mean,relative_naive_error_std]#,
                  #relative_naive_error_kde]
#[relative_naive_error,relative_naive_error_mean,relative_naive_error_std,kde,eml,ks_test,ks_norm][:-1]