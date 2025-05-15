import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import time
dtype_tf=tf.float32
dtype_np=np.float32

def evaluate_net_all_replicates_seq_snps(dna_vae,allele_data,sample_size,generations,num_replicates, round_decimals,
                                         num_train_generations,num_train_reps,position_data,save_path='',idx_list=[]):
    max_distance=1500

    num_snps = allele_data.shape[0]
    eps = 0.001

    allele_data= np.tile(allele_data.reshape((num_snps, 1,num_replicates, generations))
                         ,[1,sample_size//num_train_reps,1,1])

    z_dim=z_dim_enc=dna_vae.end_z_dim



    encoder_output=np.zeros((1,num_snps,sample_size,z_dim))
    gauss_output = np.zeros((1,num_snps,sample_size, z_dim))
    #window_out = np.zeros((1, num_snps,sample_size, z_dim))
    if dna_vae.enc_type=='sim':
        window_out = np.zeros((1,len(idx_list), sample_size, 1+(dna_vae.snp_window_half*2)))
    else:
        window_out = np.zeros((1, len(idx_list), sample_size, z_dim))


    sample_size=allele_data.shape[1]*num_train_reps#num_replicates


    #reshape + transpose input:
    allele_data=np.reshape(allele_data, ( num_snps,sample_size,generations))
    allele_data=allele_data.transpose((2,1,0))

    allele_data=np.concatenate([np.zeros((generations,sample_size,dna_vae.snp_window_half)),allele_data,np.zeros((generations,sample_size,dna_vae.snp_window_half))],axis=2)
    allele_data_predicted=np.zeros((generations-dna_vae.time_batch_size,sample_size,num_snps+2*dna_vae.snp_window_half))#
    window_len=dna_vae.snp_window_half*2+1

    idx_list_numpy=np.asarray(idx_list)


    sample_set=np.asarray(list(range(sample_size)),dtype=np.int32)
    start_time=time.time()
    for i in range(generations - dna_vae.time_batch_size):
        print('read generation',i,num_snps,(time.time()-start_time)//60,flush=True)
        start_time=time.time()
        generation_result_data=np.zeros((sample_size,num_snps+2*dna_vae.snp_window_half))
        #out_data_all=allele_data[i+dna_vae.time_batch_size]
        if i <= num_train_generations - dna_vae.time_batch_size:
            print('use train')
            generation_data=allele_data[i:i+dna_vae.time_batch_size]
        else:
            print('use own pred')
            generation_data= np.concatenate([generation_data[1:], np.expand_dims(allele_data_predicted[i-1], axis=0)], axis=0)
        for snp_idx in range(dna_vae.snp_window_half,num_snps+dna_vae.snp_window_half):
            if snp_idx%100000==0:
                print(snp_idx,(time.time()-start_time)//60,flush=True)
            #out_data=out_data_all[:,snp_idx]
            snp_input= generation_data[:,:,snp_idx].T
            start=snp_idx-dna_vae.snp_window_half
            end=(snp_idx+dna_vae.snp_window_half)+1
            neighbor_data=generation_data[:,:,start:end]
            start_p,end_p=max(snp_idx-2*dna_vae.snp_window_half,0),min(num_snps,snp_idx+1)
            position_data_tmp = position_data[start_p:end_p]
            abs_distance = np.abs(position_data[snp_idx-dna_vae.snp_window_half] - position_data_tmp)
            weight = (max_distance - abs_distance) / max_distance
            weight[weight < 0] = 0.0

            if not start_p and len(weight)!=window_len:
                weight=np.concatenate([np.zeros(window_len-len(weight)),weight],axis=0)
            elif end_p== num_snps and len(weight)!=window_len:
                weight=np.concatenate([weight,np.zeros(window_len-len(weight))],axis=0)



            snp_prediction,gauss_out,enc_out,win_out=dna_vae.eval(snp_input,neighbor_data,weight)
            snp_prediction, gauss_out, enc_out=snp_prediction.numpy()[:, 0],gauss_out.numpy(),enc_out.numpy()

            poly_positions = np.where( np.logical_and(0 != snp_input[:,-1], snp_input[:,-1] != 1))[0]
            fix_positions=np.setdiff1d(sample_set,poly_positions)
            if poly_positions.shape[0]:
                generation_result_data[poly_positions,[snp_idx]*poly_positions.shape[0]]=snp_prediction[poly_positions]
            if fix_positions.shape[0]:
                generation_result_data[fix_positions, [snp_idx]*fix_positions.shape[0]] = snp_input[fix_positions,-1]
            if not i:
                encoder_output[i,snp_idx-dna_vae.snp_window_half]=enc_out#enc_out[0]
                gauss_output[i,snp_idx-dna_vae.snp_window_half]=gauss_out
                if dna_vae.integrate_gene:
                    #1, num_snps, sample_size, 1+(dna_vae.snp_window_half*2)
                    if snp_idx-dna_vae.snp_window_half in idx_list:
                        snp_idx=idx_list.index(snp_idx-dna_vae.snp_window_half)
                        window_out[i,snp_idx]=win_out.numpy()#[0:1]#win_out[0]
        allele_data_predicted[i]=generation_result_data


    prediction=allele_data_predicted[:,:,dna_vae.snp_window_half:-dna_vae.snp_window_half]

    # fitlter the returned output based
    prediction, gauss_output, encoder_output, window_out=prediction[:,:,idx_list],gauss_output[:,idx_list],encoder_output[:,idx_list],window_out#[:,idx_list]
    return prediction,gauss_output,encoder_output,window_out

def evaluate_net_all_replicates_seq_snps_chrom(dna_vae,allele_data,sample_size,generations,num_replicates,
                                               round_decimals, num_train_generations,num_train_reps,position_data,save_path,idx_list=[]):
    prediction, gauss_output, encoder_output, window_out=dict(),dict(),dict(),dict()
    for key in sorted(allele_data.keys()):
        print('eval chrom:',key)
        chrom_allele_data=np.squeeze(allele_data[key])
        chrom_positions=np.squeeze(position_data[key])
        cprediction, cgauss_output,\
            cencoder_output, cwindow_out=evaluate_net_all_replicates_seq_snps(dna_vae,
                                         chrom_allele_data,sample_size,generations,num_replicates,
                                         round_decimals, num_train_generations,num_train_reps,chrom_positions,idx_list=idx_list)

        prediction[key]=cprediction
        gauss_output[key]=cgauss_output
        encoder_output[key]=cencoder_output
        window_out[key]=cwindow_out



    return prediction,gauss_output,encoder_output,window_out


def evaluate_net_all_replicates_in_net(dna_vae,allele_data,sample_size,generations,num_replicates, round_decimals, num_train_generations,num_train_reps,position_data):
    mini_batch=10
    add_samples=sample_size//num_train_reps
    #allele_data=allele_data[:33]
    max_distance=1500

    num_snps = allele_data.shape[0]
    eps = 0.001
    std_org=np.std(allele_data,axis=1).T
    mean_org=np.mean(allele_data,axis=1).T
    allele_data_expand= np.tile(allele_data.reshape((num_snps, 1,num_replicates, generations))
                         ,[1,add_samples,1,1])

    z_dim=z_dim_enc=dna_vae.end_z_dim
    encoder_output=np.zeros((generations-dna_vae.time_batch_size,num_snps,z_dim))
    gauss_output = np.zeros((generations  -dna_vae.time_batch_size,num_snps, z_dim))
    window_out=np.zeros((generations-dna_vae.time_batch_size,num_snps, z_dim))
    sample_size=allele_data_expand.shape[1]*num_train_reps#num_replicates


    allele_data_expand=allele_data_expand.transpose((3,0,1,2))
    zero_shape=(generations,dna_vae.snp_window_half,allele_data_expand.shape[2],allele_data_expand.shape[3])
    allele_data_expand=np.concatenate([np.zeros(zero_shape),allele_data_expand,np.zeros(zero_shape)],axis=1)
    allele_data_predicted=np.zeros((generations-dna_vae.time_batch_size,num_snps+2*dna_vae.snp_window_half,add_samples,num_replicates))#
    window_len=dna_vae.snp_window_half*2+1

    start_time=time.time()
    for i in range(generations - dna_vae.time_batch_size):
        print('read generation',i,num_snps,(time.time()-start_time)//60,flush=True)
        start_time=time.time()
        generation_result_data=np.zeros((num_snps+2*dna_vae.snp_window_half,add_samples,num_replicates))
        #out_data_all=allele_data[i+dna_vae.time_batch_size]
        if i <= num_train_generations - dna_vae.time_batch_size:
            generation_data=allele_data_expand[i:i+dna_vae.time_batch_size]
        else:
            generation_data= np.concatenate([generation_data[1:], np.expand_dims(allele_data_predicted[i-1], axis=0)], axis=0)
        for snp_idx in range(dna_vae.snp_window_half,num_snps+dna_vae.snp_window_half,mini_batch):
            end=min(num_snps+dna_vae.snp_window_half,snp_idx+mini_batch)
            multiple_spns=list(range(snp_idx,end))
            multiple_snp_len=len(multiple_spns)


            if snp_idx%50000==0:
                print(snp_idx,(time.time()-start_time)//60,flush=True)

            weight=None

            tmp=generation_data[:,multiple_spns]

            snp_input = np.transpose(tmp, (1, 2, 3, 0))
            snp_input_concat=np.concatenate(snp_input,axis=0)

            neighbors=[]
            for e_idx in multiple_spns:
                start=e_idx-dna_vae.snp_window_half
                end=(e_idx+dna_vae.snp_window_half)+1
                neighbor_data=np.transpose(generation_data[:,start:end],(2,1,0,3))
                neighbors.append(neighbor_data)
            neighbors=np.concatenate(neighbors)



            snp_prediction,gauss_out,enc_out,win_out=dna_vae.eval_parallel(snp_input_concat,neighbors,weight)

            poly_positions = np.where( np.logical_and(0 != snp_input[:,:,:,-1], snp_input[:,:,:,-1] != 1))#[0]

            generation_result_data[multiple_spns]=snp_input[:,:,:,-1]#.T

            if poly_positions[0].shape[0]:
                snp_prediction=snp_prediction.reshape((multiple_snp_len,add_samples,num_replicates))
                generation_result_data[poly_positions[0], poly_positions[1], poly_positions[2]] = \
                snp_prediction[poly_positions[0],poly_positions[1], poly_positions[0]]
            multiple_spns=[ entry-dna_vae.snp_window_half for entry in multiple_spns]
            encoder_output[i,multiple_spns]=np.mean(enc_out.reshape((multiple_snp_len,add_samples,dna_vae.end_z_dim)),axis=1)
            gauss_output[i,multiple_spns]=np.mean(gauss_out.reshape((multiple_snp_len,add_samples,dna_vae.end_z_dim)),axis=1)
            window_out[i,multiple_spns]=np.mean(win_out.reshape((multiple_snp_len,add_samples,dna_vae.end_z_dim)),axis=1)
        allele_data_predicted[i]=generation_result_data

    prediction=allele_data_predicted[:,dna_vae.snp_window_half:-dna_vae.snp_window_half]
    prediction=np.reshape(prediction,(prediction.shape[0],prediction.shape[1],prediction.shape[2]*prediction.shape[3]))
    prediction=np.transpose(prediction,(0,2,1))

    std_pred = np.std(prediction, axis=1)
    mean_pred=np.mean(prediction, axis=1)

    subtracted_mean=np.abs(mean_org[dna_vae.time_batch_size:] - mean_pred)
    substracted_std=np.abs(std_org[dna_vae.time_batch_size:]-std_pred)


    return prediction,subtracted_mean,substracted_std,gauss_output,encoder_output,window_out

