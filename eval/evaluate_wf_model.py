import os
import numpy as np
import pickle

from wf_model.unvariate_models import simulate

def chrom_eval(allele_data,num_replicates,tracing_generations,sample_size,positions,estimated_s,num_train_generations,Ne,n_sampling,n_census,coverage):
    frequencies=np.copy(allele_data)
    snp_counter=frequencies.shape[0]
    print('!!!!DEBUG_shapes', frequencies.shape)
    frequencies=np.transpose(frequencies,(1,0,2))
    print('!!!!DEBUG_shapes', frequencies.shape)
    eps=0.001

    snp_freqs_wf=np.zeros((num_replicates,len(tracing_generations)+1,sample_size,snp_counter))

    frequencies = np.transpose(frequencies, (0, 1, 2))
    print('!!!!DEBUG_shapes', frequencies.shape)

    std_org = np.transpose(np.std(frequencies, axis=0),(1,0))[1:]#,1:
    mean_org = np.transpose(np.mean(frequencies, axis=0),(1,0))[1:]
    print('!!!!DEBUG_shapes',frequencies.shape,std_org.shape,Ne)

    for rep in range(num_replicates):
        freqs=[]

        for i,(all_gen_freqs,position) in enumerate(zip(frequencies[rep],positions)):

            if estimated_s!=None:
                if position in estimated_s:
                    s=estimated_s[position]
                else:
                    s = 0
                    print('Oh no!!! There are some crazy points',position,flush=True)
            else:
                s=0

            all_freqs=[]
            for step_counter in range(num_train_generations-1):#range(num_train_generations-2):
                start_freq = all_gen_freqs[step_counter]
                measured_freqs = simulate(start_freq, Ne, [tracing_generations[0]], sample_size, n_round=8,h=0.5,s=s,n_sampling=n_sampling,n_census=n_census,coverage=coverage)
                measured_freqs=np.asarray(measured_freqs)#[1:]
                if not step_counter:
                    all_freqs.append(measured_freqs[0])

                all_freqs.append(measured_freqs[1])
            start_freq = all_gen_freqs[num_train_generations-1]#all_gen_freqs[num_train_generations-1]
            tracing_generations_new = [tracing_generations[0] * count for count in
                                       range(1, len(tracing_generations) - num_train_generations + 2)]

            measured_freqs = simulate(start_freq, Ne, tracing_generations_new, sample_size, n_round=8, h=0.5,
                                      s=s)
            for frec_count in range(1,len(measured_freqs)):
                all_freqs.append(measured_freqs[frec_count])

            measured_freqs=np.stack(all_freqs)
            freqs.append(measured_freqs)
            all_gen_freqs[1:][all_gen_freqs[1:] < 1] = 0
            #TODO track NaN values
        freqs=np.asarray(freqs)
        freqs=np.transpose(freqs,(1,2,0))
        snp_freqs_wf[rep]=freqs


    std_pred = np.std(snp_freqs_wf, axis=(0,2))[1:]

    all_stds=np.abs(std_pred-std_org)

    mean_pred = np.mean(snp_freqs_wf, axis=(0,2))[1:]
    all_abs_means_=np.abs(mean_org-mean_pred)
    return snp_freqs_wf,mean_org,std_org,all_abs_means_,all_stds

def eval_wf_model(WF_out_path,generations,num_replicates,step,allele_data,num_train_generations,
                  sample_size,estimated_s,Ne,positions,start_generation_idx=0,n_sampling=1000,n_census=1000,coverage=None,idx_list=[]):
    if os.path.isfile(WF_out_path):
        return
    print('eval wf')
    out_stream=open(WF_out_path,'wb')
    tracing_generations=[i*step for i in range(start_generation_idx+1,generations)]
    snp_freqs_wf,mean_org,std_org,all_abs_means_,all_stds=chrom_eval(allele_data, num_replicates,
                                                                     tracing_generations, sample_size, positions, estimated_s,
                                                                     num_train_generations, Ne, n_sampling, n_census, coverage)

    save_object={'freqs':snp_freqs_wf[:,:,:,idx_list]}
    pickle.dump(save_object,out_stream)
    out_stream.close()

    return None,None,snp_freqs_wf,None,mean_org,std_org,all_abs_means_,all_stds,all_abs_means_


def eval_wf_model_chroms(WF_out_path,generations,num_replicates,step,allele_data,num_train_generations,
                  sample_size,estimated_s,Nes,positions,start_generation_idx=0,n_sampling=1000,n_census=1000,coverage=None,idx_list=[]):
    if os.path.isfile(WF_out_path):
        return
    print('eval wf')
    out_stream=open(WF_out_path,'wb')
    tracing_generations=[i*step for i in range(start_generation_idx+1,generations)]
    snp_freqs_wf=dict()

    for key in sorted(allele_data.keys()):
        chrom_allele_data=np.squeeze(allele_data[key])
        chrom_estimated_s=estimated_s[key]
        chrom_positions=np.squeeze(positions[key])

        tmp=chrom_allele_data[chrom_allele_data>1]
        if len(tmp):
            print(tmp.shape)
            print(tmp)

        print('DEBUG shape',key,chrom_allele_data.shape)
        Ne=Nes#[key]

        csnp_freqs_wf,cmean_org,cstd_org,call_abs_means_,call_stds=chrom_eval(chrom_allele_data, num_replicates,
                                                                         tracing_generations, sample_size, chrom_positions, chrom_estimated_s,
                                                                         num_train_generations, Ne, n_sampling, n_census, coverage)
        snp_freqs_wf[key]=csnp_freqs_wf[:,:,:,idx_list]


    save_object={'freqs':snp_freqs_wf}
    pickle.dump(save_object,out_stream)
    out_stream.close()

    return None,None,snp_freqs_wf,None
