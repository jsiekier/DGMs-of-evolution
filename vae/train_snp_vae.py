import shutil
from munch import DefaultMunch
import os
import json
import time
import tensorflow as tf

from data_processing.Data_loader import Dataset_loader_VAE_bi_allelic, DS_loader_bi_allelic_replicates, \
    Dataset_loader_VAE_bi_allelic_debug

tf.keras.backend.set_floatx('float32')
import random
import re


from vae.snp_vae import DNA_VAE
from variables import fix_variables
import  numpy as np
random.seed(0)





def get_fix_variable_vals(animal):

    num_generations=fix_variables[animal]['num_generations']
    num_replicates=fix_variables[animal]['num_replicates']
    MAX_GENE_LENGTH = fix_variables[animal]["max_gene_len"]
    return num_generations,num_replicates,MAX_GENE_LENGTH

loss_names=['loss','rec_loss','kld','entropy','l1','MSE','SF MSE']

def print_loss(losses, train_step_time,stopping_counter,title,step,summary_writer,z_reps_mean,z_reps_std,beta,metadata):
    # all_loss,entro_loss ,kl_loss, l1_loss

    print(title,end=':')

    for name,loss in zip(loss_names,losses):
        print(name,round(loss,5),sep=': ',end=' - ')
    print('mean:',end='[')
    for var in z_reps_mean:
        print(var,end=';')

    print('] std:',end='[')
    for var in z_reps_std:
        print(var,end=';')


    if summary_writer!=None:
        with summary_writer.as_default():
            for name, loss in zip(loss_names, losses):
                tf.summary.scalar('statistics/'+name, loss, step=step)

    print('] KLD:',beta)
    print('step: '+str(step)+', stopping counter: '+str(stopping_counter)+' time: '+str(round(train_step_time,2)),flush=True)


def load_dna_cnn(args,MAX_GENE_LENGTH,NUM_INIT_COLOR_CHANELS,checkpoint_path=None,model_part=''):
    dna_cnn = DNA_VAE(activation=args.non_linearity,
                      max_gene_length=MAX_GENE_LENGTH,
                      num_init_color_chanels=NUM_INIT_COLOR_CHANELS,
                      num_layers=args.num_layers,
                      dilations=args.dilations,
                      filter_size=args.filter_size,
                      architecture=args.nn,
                      batchnorm_axis=args.batchnorm_axis,
                      batchnorm_renorm=args.batchnorm_renorm,
                      dropout=args.dropout,
                      gauss_lay=args.gauss_lay,
                      gauss_std=args.gauss_std,
                      learning_rate=args.learning_rate,
                      opt=args.opt, use_resnet=args.use_resnet,
                      time_batch_size=args.time_batch_size,
                      use_map=args.use_map,
                      end_z_dim=args.end_z_dim,
                      reduce_shortcut=args.reduce_shortcut,
                      round_decimals=args.round_decimal,
                      recursive_enc=args.recursive_enc,
                      recursive_dec=args.recursive_dec,
                      num_snp_layers=args.num_snp_layers,
                      snp_window_half=args.num_neighbors,
                      integrate_gene=args.integrate_gene,
                      layer_dims=args.layer_dims,
                      position_info=args.position_info,
                      rec_loss=args.rec_loss,
                      checkpoint_path=checkpoint_path,
                      use_selection_coefficient=args.use_selection_coefficient,
                      include_parallel_info=args.include_parallel_info,
                      prior=args.prior,
                      linkage_prior=args.linkage_prior,
                      enc_type=args.enc_type,
                      loss_type=args.loss_type,
                      exclude_sf=args.exclude_sf,
                      sf_ex_alpha=args.sf_ex_alpha,
                      model_part=model_part,
                      debug=args.debug,
                      multiply_result=args.multiply_result
                      )
    return dna_cnn


def main_training(args,use_tensorboard=False,tmp_data_folder='',kld_influence=0.0001,loading_name='',model_part=''):
    print('train model',args)
    fine_tuning=args.fine_tune
    if fine_tuning:

        model_path=tmp_data_folder+"models/" +args.nn_name+'_fine'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print('create', model_path, flush=True)
        shutil.copyfile(model_path[:-5] + "/output.json", model_path + "/output.json")
        json_file = open(model_path + "/output.json", 'r')
        json_data = json.load(json_file)

        learning_rate=args.learning_rate
        args=DefaultMunch.fromDict(json_data)
        args.learning_rate=learning_rate

    else:
        model_path=tmp_data_folder+"models/" +args.nn_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print('create', model_path, flush=True)
        output_1 = model_path + "/output.json"
        out_1 = open(output_1, 'w')
        json.dump(vars(args), out_1)
        out_1.close()
        print('save params')


    animal = args.animal
    BATCHSIZE=args.batch_size
    NUM_INIT_COLOR_CHANELS=args.num_init_color_chanels
    train_sessions=args.train_sessions
    repeat_iterations=args.repeat_iterations
    print_every=args.print_every
    work_from=args.work_from


    num_generations, num_replicates, MAX_GENE_LENGTH=get_fix_variable_vals(animal)


    train_summary_writer=None

    if use_tensorboard:
        train_log_dir = model_path + '/logs/'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir + 'train')

    print(animal,flush=True)
    print(fix_variables[animal],flush=True)
    print(fix_variables[animal]["num_train_generations"],flush=True)
    train_generations = fix_variables[animal]["num_train_generations"][args.num_train_it]
    num_train_reps=fix_variables[animal]['num_train_reps'][args.num_replicates]
    tmp=fix_variables[work_from][animal]

    input_args={'sync_file':fix_variables[work_from][animal]["sync_process"],
                                         'num_init_color_channels':NUM_INIT_COLOR_CHANELS,
                                         'decimals':args.round_decimal,
                                         'num_generations':num_generations,
                                         'num_replicates':num_replicates,
                                        'num_train_generations':train_generations,
                                        'num_neighbors':args.num_neighbors,
                                         'num_train_reps':num_train_reps,
                                        'max_dist_neighbor':None,
                                         'fine_tune':args.fine_tune}
    if args.include_parallel_info:
        dataset_train = DS_loader_bi_allelic_replicates(**input_args)
    else:
        if args.debug:
            dataset_train = Dataset_loader_VAE_bi_allelic_debug(**input_args)
        else:
            dataset_train = Dataset_loader_VAE_bi_allelic(**input_args)

    gene_loader_train = dataset_train.batch(BATCHSIZE).prefetch(tf.data.experimental.AUTOTUNE)

    if fine_tuning:
        if loading_name == '':
            checkpoint_path =model_path[:-5] + '/tf_ckpts_last'
        else:
            checkpoint_path = tmp_data_folder + "models/" + loading_name + '/tf_ckpts_last'
    else:
        if loading_name=='':
            checkpoint_path=model_path + '/tf_ckpts_last'
            if args.nn=="VAE":
                checkpoint_path=checkpoint_path.replace("VAE","AE")
                checkpoint_path=re.sub("_KLD[1-9]", "",checkpoint_path)

        else:
            checkpoint_path = tmp_data_folder+"models/" +loading_name+ '/tf_ckpts_last'
    dna_cnn = load_dna_cnn(args, MAX_GENE_LENGTH, NUM_INIT_COLOR_CHANELS, checkpoint_path,model_part)#
    print('Model creation finished',flush=True)
    ckpt_last = tf.train.Checkpoint(step=tf.Variable(1), optimizer=dna_cnn.optimizer, net=dna_cnn)
    manager_last = tf.train.CheckpointManager(ckpt_last, model_path + '/tf_ckpts_last', max_to_keep=2)

    early_stop_counter=0
    curr_loss=1000000
    #rising_factor=kld_influence
    #beta=rising_factor

    start_time = time.time()
    algo_time = time.time()

    KLD_anneal=args.KLD_anneal
    KLD_vals=np.concatenate([np.linspace(kld_influence,kld_influence*10,10,dtype=np.float32),np.asarray([kld_influence*10]*2,dtype=np.float32)],axis=0)



    for i in range(train_sessions):
        if KLD_anneal:
            curr_KLD_influence = KLD_vals[i%KLD_vals.shape[0]]
        else:
            curr_KLD_influence=kld_influence
        if ((time.time()-algo_time)/3600>23):
            print('time over',(time.time()-algo_time)/3600)
            break


        ckpt_last.step.assign_add(1)


        train_losses,z_reps_mean,z_reps_std,metadata= dna_cnn.train_iterations_lstm(gene_loader_train, train_generations, beta=curr_KLD_influence)

        if i % print_every == 0:

            print_loss(train_losses,
                       (time.time() - start_time) / 60,
                       early_stop_counter,
                       'train',i,train_summary_writer,z_reps_mean,z_reps_std,beta=curr_KLD_influence,metadata=metadata)
            print('attention linkage correlation',metadata)
            start_time = time.time()
            if np.isnan(train_losses[0]):
                print('load old checkpoint')
                ckpt_last.restore(manager_last.latest_checkpoint)

            else:
                manager_last.save()

                if train_losses[0] < curr_loss:
                    curr_loss = train_losses[0]
                    early_stop_counter = 0

                else:
                    early_stop_counter += 1



