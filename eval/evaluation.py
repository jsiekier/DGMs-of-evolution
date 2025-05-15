import json
import os
import numpy as np
import pickle
import random

from eval.evaluate_wf_model import eval_wf_model, eval_wf_model_chroms
from vae.snp_vae import DNA_VAE

random.seed(0)
from eval.eval_functions import evaluate_net_all_replicates_seq_snps, evaluate_net_all_replicates_in_net, \
    evaluate_net_all_replicates_seq_snps_chrom
from variables import fix_variables


dtype=np.float32

def read_estimated_s(estimated_s_path):
    base_to_idx = {'a': 0, 'A': 0, 't': 1, 'T': 1, 'c': 2, 'C': 2, 'g': 3, 'G': 3}

    # check if path is a file or directory:
    estimated_s_files=[]
    if os.path.isdir(estimated_s_path):
        for f in os.listdir(estimated_s_path):
            estimated_s_files.append(estimated_s_path+f)
    else:
        estimated_s_files=[estimated_s_path]
    estimated_s=dict()
    for estimated_s_file in estimated_s_files:
        haplo_stream = open(estimated_s_file, 'r')

        i=0
        chrom=None
        for line in haplo_stream:
            i+=1
            k = line.replace('\n','').split(" ")
            if k[0]=="NA" or k[0]=='nan':
                s=0
            else:
                # take the negative value as pool-seq computes s for the minor allele and we look at the major allele
                s=float(k[0])
                if s <-1 or s>1:
                    s=0

            chrom=k[1]
            #minor_nucleotide=k[3]

            position=int(k[2])#-1
            if chrom not in estimated_s:
                estimated_s[chrom]=dict()
            estimated_s[chrom][position]=s#,base_to_idx[minor_nucleotide]]
        print('num lines:',i)
    return estimated_s


def load_net(net_name, model_type, tmp_data_folder, args=None):
    if args == None or not args['fine_tune']:
        model_folder = tmp_data_folder + 'models/' + net_name
    else:
        model_folder = tmp_data_folder + 'models/' + net_name + '_fine'
    model_param_path = model_folder + '/output.json'

    if model_type == 'latest':
        checkpoint_end = '/tf_ckpts_last'
    elif model_type == 'train':
        checkpoint_end = '/tf_ckpts_train'
    elif model_type == 'val':
        checkpoint_end = '/tf_ckpts'
    else:
        print('ERROR!!! model type not valid')

    if os.path.isfile(model_param_path):

        json_file = open(model_param_path, 'r')
        json_data = json.load(json_file)
        args = json_data
        checkpoint_path = model_folder + checkpoint_end
    elif args != None:
        print('ATTENTION RANDOM MODEL USED')
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        checkpoint_path = None
    else:
        print('no/wrong model information given!!!')

    animal = args['animal']
    MAX_GENE_LENGTH = fix_variables[animal]["max_gene_len"]
    num_train_reps = fix_variables[animal]['num_train_reps'][args['num_replicates']]
    num_train_generations = fix_variables[animal]['num_train_generations'][args['num_train_it']]

    #for _ in range(num_replicates):
    dna_cnn_actor = DNA_VAE(activation=args['non_linearity'],
                     max_gene_length=MAX_GENE_LENGTH,
                     num_init_color_chanels=args['num_init_color_chanels'],
                     num_layers=args['num_layers'],
                     dilations=args['dilations'],
                     filter_size=args['filter_size'],
                     architecture=args['nn'],
                     batchnorm_axis=args['batchnorm_axis'],
                     batchnorm_renorm=args['batchnorm_renorm'],
                     dropout=args['dropout'],
                     gauss_lay=args['gauss_lay'],
                     gauss_std=args['gauss_std'],
                     opt=args['opt'],
                     learning_rate=args['learning_rate'],
                     time_batch_size=args['time_batch_size'],
                     use_resnet=args['use_resnet'],
                     use_map=args['use_map'],
                     checkpoint_path= checkpoint_path,
                     end_z_dim=args['end_z_dim'],
                     reduce_shortcut=args['reduce_shortcut'],
                     round_decimals=args['round_decimal'],
                     recursive_enc=args['recursive_enc'],
                     recursive_dec=args['recursive_dec'],
                     num_snp_layers=args['num_snp_layers'],
                     snp_window_half=args['num_neighbors'],
                     integrate_gene=args['integrate_gene'],
                     position_info=args['position_info'],
                     layer_dims=args['layer_dims'],
                     rec_loss=args['rec_loss'],
                     use_selection_coefficient=args['use_selection_coefficient'],
                     include_parallel_info=args['include_parallel_info'],
                     prior=args['prior'],
                     linkage_prior=args['linkage_prior'],
                     enc_type=args['enc_type'],
                     exclude_sf=args['exclude_sf'],
                     multiply_result=args['multiply_result'] if 'multiply_result' in args else 0
                            )

        #dna_cnn_actors.append(dna_cnn_actor)
    return animal, MAX_GENE_LENGTH, dna_cnn_actor,args['round_decimal'],num_train_reps,args['num_replicates'],num_train_generations,args['num_train_it']





def predict_polymorphic_values_snp_reps(folder_name, animal, MAX_GENE_LENGTH, dna_cnn, round_decimals, sample_size,work_from,
                               start_generation_idx,save_path,wf_save_path, num_unselected_snps, num_selected_snps_region
                                        ,use_all_train_data,num_train_reps,num_train_generations):
    # load eval indices for later filtering:
    eval_path = fix_variables[work_from][animal]['eval_positions']
    print(eval_path)
    linkage_data = pickle.load(open(eval_path, 'rb'))
    idx_list = np.asarray(linkage_data['selected']).tolist() + np.asarray(linkage_data['unselected']).tolist()

    generations=fix_variables[animal]['num_generations']
    num_replicates=fix_variables[animal]['num_replicates']

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sync_file=fix_variables[work_from][animal]["sync_process"]

    data = pickle.load(open(sync_file, 'rb'))
    allele_data, position_data, fine_tuning_indices = data['alleles'], data['distances'], data['fine_tuning_indices']


    if dna_cnn.include_parallel_info:
        prediction, subtracted_mean, substracted_std,\
        gauss_output, encoder_output,window_output= evaluate_net_all_replicates_in_net(dna_cnn,allele_data,sample_size,
                                                                       generations,
                                                                       num_replicates, round_decimals,
                                                                       num_train_generations,
                                                                       num_train_reps,position_data)

    else:
        if isinstance(allele_data, dict):
            prediction, gauss_output, encoder_output,window_output= evaluate_net_all_replicates_seq_snps_chrom(dna_cnn,allele_data,sample_size,
                                                                           generations,
                                                                           num_replicates, round_decimals,
                                                                           num_train_generations,
                                                                           num_train_reps,position_data,save_path+'/'+folder_name,
                                                                                                               idx_list=idx_list)
        else:
            prediction,gauss_output, \
            encoder_output,window_output= evaluate_net_all_replicates_seq_snps(dna_cnn,allele_data,sample_size,
                                                                           generations,
                                                                           num_replicates, round_decimals,
                                                                           num_train_generations,
                                                                           num_train_reps,position_data,save_path=save_path+'/'+folder_name,
                                                                               idx_list=idx_list)

    save_object={'freqs':prediction,'all_z_tide':encoder_output,
                 'all_z_samples':gauss_output}
    if dna_cnn.integrate_gene:
        save_object['window_representation']=window_output
    pickle.dump(save_object,open(save_path+'/'+folder_name+'.pkl','wb'))


    

def evaluate_wf_model(wf_save_path,animal,work_from='mogon',sample_size=100):
    # load eval indices for later filtering:
    generations = fix_variables[animal]['num_generations']
    num_replicates = fix_variables[animal]['num_replicates']
    step = fix_variables[animal]['step_size']
    sync_file = fix_variables[work_from][animal]["sync_process"]
    eval_path = fix_variables[work_from][animal]['eval_positions']
    num_train_generations=fix_variables[animal]['num_train_generations'][0]
    num_train_reps = fix_variables[animal]['num_train_reps'][0]

    print(eval_path)
    linkage_data = pickle.load(open(eval_path, 'rb'))
    idx_list = np.asarray(linkage_data['selected']).tolist() + np.asarray(linkage_data['unselected']).tolist()

    data = pickle.load(open(sync_file, 'rb'))
    allele_data, position_data, fine_tuning_indices = data['alleles'], data['distances'], data['fine_tuning_indices']

    if 'estimated_s' in fix_variables[work_from][animal]:
        estimated_s_file=fix_variables[work_from][animal]['estimated_s'][0][0]
        if estimated_s_file != '':
            print(estimated_s_file)
            estimated_s = read_estimated_s(estimated_s_file)
    else:
        estimated_s=None

    if num_train_generations==11:
        Ne=fix_variables[animal]['Ne'][1]
    elif num_train_generations==7:
        Ne = fix_variables[animal]['Ne'][0]
    else:
        Ne = fix_variables[animal]['Ne']

    n_sampling=fix_variables[animal]['Nsampling']
    n_census = fix_variables[animal]['Ncensus']
    coverage=fix_variables[animal]['coverage']
    #'''
    if isinstance(allele_data,dict):
        print('eval on chronms')
        eval_wf_model_chroms(wf_save_path,generations,num_replicates,step,allele_data,num_train_generations,
                      sample_size//num_train_reps,estimated_s,Ne,position_data,n_sampling=n_sampling,n_census=n_census,coverage=coverage,idx_list=idx_list)
    else:
        print('shape allele data:', allele_data.shape[0])
        eval_wf_model(wf_save_path,generations,num_replicates,step,allele_data,num_train_generations,
                      sample_size//num_train_reps,list(estimated_s.values())[0],
                      Ne,position_data,n_sampling=n_sampling,n_census=n_census,coverage=coverage,idx_list=idx_list)



def evaluate_drift_samples(work_from, model_name,model_type,sample_size,num_unselected_snps,num_selected_snps_region
                           ,tmp_data_folder='',wf_file_name='',use_all_train_data=False,args=None):

    if args['fine_tune']:
        model_folder = tmp_data_folder+'models/'+model_name+'_fine/eval_data'
    else:
        model_folder = tmp_data_folder+'models/'+model_name+'/eval_data'
    print(model_folder)


    #load network
    animal, MAX_GENE_LENGTH, dna_cnn, round_decimals,num_train_reps,num_train_reps_idx,num_train_generations,num_train_generations_idx= load_net(model_name,
                                                                                   model_type,
                                                                                   tmp_data_folder,args=args)
    #if not recursive:
    # if the start freqs of all replicates are equal
    predict_polymorphic_values_snp_reps('train_data',animal, MAX_GENE_LENGTH, dna_cnn, round_decimals,sample_size,work_from,
                               start_generation_idx=0,save_path=model_folder,
                               wf_save_path=wf_file_name, num_unselected_snps=num_unselected_snps,
               num_selected_snps_region = num_selected_snps_region,use_all_train_data=use_all_train_data,
                               num_train_reps=num_train_reps,num_train_generations=num_train_generations)
