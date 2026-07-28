import tensorflow as tf
import numpy as np
from scipy import stats

from vae.layer_functions import decoder_mlp_interpretable, make_focus_snp_encoder, make_window_CNN


tf.keras.backend.set_floatx('float32')


dtype=tf.float32


def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - tf.math.log(1. / x - 1.)

class Seq_n(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, batchnorm, activation, num_layers, activation_last_layer):
        super(Seq_n, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(tf.keras.layers.Dense(input_dim // (2 ** i), activation=activation))
            if batchnorm:
                layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dense(latent_dim, activation=activation_last_layer))

        self.mlp = tf.keras.Sequential(layers)

    def call(self, seq):
        return self.mlp(seq)

    def call(self, seq):
        return self.mlp(seq)
class FocusSeqEnc(tf.keras.Model):
    def __init__(self, latent_dim, z_dim,num_neighbors,batchnorm,
                 similarity_measure='cosine',integrate_focus=False,num_layer=1,activation=tf.keras.activations.tanh,
                 enc1_activation='tanh',use_enc3=0,use_neg_softmax=0,data_transform='None'):
        super(FocusSeqEnc, self).__init__()
        self.num_neighbors=num_neighbors
        self.encoder1 = Seq_n(latent_dim,latent_dim//2,batchnorm,activation='relu',num_layers=1,activation_last_layer=activation)
        self.encoder2 = Seq_n(latent_dim,latent_dim//2,batchnorm,activation='relu',num_layers=1,activation_last_layer=activation)
        if data_transform=='delta':
            self.data_transform= lambda v: tf.concat([v[..., :1], v[..., :-1] - v[..., 1:]], axis=-1)
        else:
            self.data_transform = lambda x:x

        if use_enc3:
            self.focus_enc2 = Seq_n(latent_dim,latent_dim//2,batchnorm,activation='relu',num_layers=1,activation_last_layer=activation)
        else:
            self.focus_enc2=self.encoder2

        self.similarity_measure = similarity_measure
        self.integrate_focus=integrate_focus
        if use_neg_softmax==1:
            self.use_neg_softmax=lambda x:tf.keras.activations.tanh(x) * tf.nn.softmax(tf.abs(x), axis=-1)#2*tf.keras.activations.softmax(x,axis=-1)-1
        elif use_neg_softmax==0:
            self.use_neg_softmax =lambda x:tf.keras.activations.softmax(x,axis=-1)
        elif use_neg_softmax == 3:
            self.use_neg_softmax = lambda x: tf.keras.activations.tanh * tf.nn.sigmoid(tf.abs(x)) *(1/100)
        elif use_neg_softmax == 2:
            self.use_neg_softmax = lambda x: tf.keras.activations.sigmoid(x)*(1/100)

        self.output_layer = tf.keras.layers.Dense(z_dim,activation='tanh')  # For final prediction task
        print('HUHU')



    def call(self, focus_seq, neighbor_seqs):
        focus_seq_extra_dim = tf.expand_dims(focus_seq, axis=1)
        neighbor_seqs_transform=self.data_transform(neighbor_seqs)
        focus_seqs_transform =self.data_transform(focus_seq_extra_dim)

        # Encode neighbor sequences
        neighbor_enc1 = self.encoder1(neighbor_seqs_transform)  # Shape: (batch_size, n, latent_dim)
        focus_enc1 = self.encoder1(focus_seqs_transform)



        # Step 1: Normalize focus_enc2 and neighbor_enc2 along the feature dimension
        focus_enc2_normalized = tf.nn.l2_normalize(focus_enc1, axis=-1)  # Shape: (batch_size, 1, latent_dim)
        neighbor_enc2_normalized = tf.nn.l2_normalize(neighbor_enc1, axis=-1)  # Shape: (batch_size, n, latent_dim)

        # Step 1: Compute similarity scores (dot product similarity)
        similarity_scores = tf.matmul(focus_enc2_normalized, neighbor_enc2_normalized, transpose_b=True)

        # Step 2: Normalize similarity scores to [0, 1] range using softmax
        attention_weights =self.use_neg_softmax(similarity_scores)


        neighbor_enc2 = self.encoder2(neighbor_seqs)  # Shape: (batch_size, n, latent_dim)
        focus_enc2 = self.focus_enc2(focus_seq_extra_dim)

        # Step 3: Weight the neighbor representations (neighbor_enc1)
        neighbor_info = tf.matmul(attention_weights, neighbor_enc2)

        # Reshape attention_weights to (batch_size, n) for easier interpretation
        similarities = tf.squeeze(similarity_scores, axis=1)  # Shape: (batch_size, n)

        combined_representation = tf.squeeze(tf.concat([focus_enc2, neighbor_info], axis=-1))


        # Final prediction
        return self.output_layer(combined_representation),similarities,focus_enc1,attention_weights

class DNA_VAE(tf.keras.Model):

    def __init__(self, activation, max_gene_length, num_init_color_chanels, num_layers, dilations,
                 filter_size, architecture, batchnorm_axis, batchnorm_renorm, dropout, gauss_lay, gauss_std,
                 learning_rate, opt,
                 time_batch_size, use_resnet, use_map, end_z_dim, reduce_shortcut, round_decimals, recursive_enc,
                 recursive_dec,
                 num_snp_layers, snp_window_half, integrate_gene, layer_dims, position_info, rec_loss,
                 checkpoint_path=None,
                 use_selection_coefficient=False, include_parallel_info=0, prior='Gauss', linkage_prior=0,
                 enc_type='MLP'
                 , loss_type='L2', exclude_sf=False, sf_ex_alpha=1.0, model_part='', debug=0, multiply_result=0,
                 integrate_focus=True, free_bits=0.0, NLL_lambda=0.0, data_transform='None', enc1_activation='tanh',
                 use_enc3=0, use_neg_softmax=0):
        super(DNA_VAE, self).__init__()
        self.prior = prior
        self.recursive_enc = recursive_enc
        self.recursive_dec = recursive_dec
        self.time_batch_size = time_batch_size
        self.gauss_std = gauss_std
        self.max_gene_length = max_gene_length
        self.architecture = architecture
        self.num_init_color_chanels = num_init_color_chanels
        self.dropout = dropout
        self.dilations = dilations
        self.end_z_dim = end_z_dim
        self.reduce_shortcut = reduce_shortcut
        self.use_selection_coefficient = use_selection_coefficient
        self.include_parallel_info = include_parallel_info
        self.use_distance_weights = position_info
        self.exclude_sf = exclude_sf
        self.sf_ex_alpha = sf_ex_alpha
        self.model_part = model_part
        self.debug = debug
        self.free_bits = free_bits
        self.NLL_lambda = NLL_lambda

        if batchnorm_axis == -1:
            self.batchnorm_axis = 1
        if batchnorm_axis == 1:
            self.batchnorm_axis = -1
        else:
            self.batchnorm_axis = 0
        if activation == 'relu':
            self.activation = tf.keras.activations.relu
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh

        self.batchnorm_renorm = batchnorm_renorm
        self.gauss_lay = gauss_lay
        self.round = round_decimals
        # self.num_snp_layers=num_snp_layers
        self.snp_window_half = snp_window_half
        self.integrate_gene = integrate_gene
        self.use_map = use_map
        self.linkage_prior = linkage_prior
        self.loss_type = loss_type
        self.enc_type = enc_type
        self.multiply_result = multiply_result
        self.integrate_focus = integrate_focus

        self.concadenate = tf.keras.layers.Concatenate(axis=-1)
        # enc_dims = [50, 100, 100, 150, 100, 50]
        # layer_dims=[100,50,25,25]#[100,100,50,50,25,25]
        layer_dims_neighbors = [(snp_window_half * 2) + 1, snp_window_half, snp_window_half // 2,
                                snp_window_half // 4]
        end_dim = 1
        enc_dims = [100, 50, 50]
        if self.integrate_gene:

            if self.include_parallel_info:
                reps = 10

                self.rep_combiner = tf.keras.layers.Dense(self.end_z_dim, activation=self.activation,
                                                          use_bias=False,
                                                          input_shape=(-1, self.time_batch_size * 10),
                                                          name='tbs_combiner', dtype=dtype)
            else:
                reps = 1

            if enc_type == 'MLP':
                self.time_combiner = tf.keras.layers.Dense(self.end_z_dim, activation=self.activation,
                                                           use_bias=False,
                                                           input_shape=(-1, self.time_batch_size * self.end_z_dim),
                                                           name='tbs_combiner', dtype=dtype)
                self.encoder_snp_window, last_filter_num_snp, z_dim_real_snp = make_focus_snp_encoder(self,
                                                                                                      batchnorm_axis,
                                                                                                      (
                                                                                                              snp_window_half * 2) + 1,
                                                                                                      layer_dims_neighbors,
                                                                                                      encoder_name='neighbor_enc_window')

                self.z1 = tf.keras.layers.Dense(self.end_z_dim, activation=self.activation, use_bias=False,
                                                input_shape=(-1, 2 * self.end_z_dim), name='z', dtype=dtype)
                self.encoder_focus_snp, last_filter_num_snp, z_dim_real_snp = make_focus_snp_encoder(self,
                                                                                                     batchnorm_axis,
                                                                                                     time_batch_size,
                                                                                                     enc_dims,
                                                                                                     encoder_name='focus_snp_enc')
            elif enc_type == 'sim' or enc_type == 'rep_sim' or enc_type == 'sim_rep':
                # '''
                self.encoder_snp_window = FocusSeqEnc(latent_dim=32,
                                                      z_dim=self.end_z_dim,
                                                      num_neighbors=snp_window_half,
                                                      batchnorm=batchnorm_axis,
                                                      integrate_focus=integrate_focus,
                                                      num_layer=1)  # , similarity_measure=)

            else:

                self.encoder_snp_window = make_window_CNN(self, batchnorm_axis,
                                                          ((snp_window_half * 2) + 1, self.time_batch_size, reps),
                                                          encoder_name='neighbor_enc_window_cnn')

                self.z1 = tf.keras.layers.Dense(self.end_z_dim, activation=self.activation, use_bias=False,
                                                input_shape=(-1, 2 * self.end_z_dim), name='z', dtype=dtype)
                self.encoder_focus_snp, last_filter_num_snp, z_dim_real_snp = make_focus_snp_encoder(self,
                                                                                                     batchnorm_axis,
                                                                                                     time_batch_size,
                                                                                                     enc_dims,
                                                                                                     encoder_name='focus_snp_enc')
        else:
            self.encoder_focus_snp, last_filter_num_snp, z_dim_real_snp = make_focus_snp_encoder(self,
                                                                                                 batchnorm_axis,
                                                                                                 time_batch_size,
                                                                                                 enc_dims,
                                                                                                 encoder_name='focus_snp_enc')

        self.loss_idx = 2
        if rec_loss == 'entropy':
            self.loss_idx = 0
        elif rec_loss == 'l1':
            self.loss_idx = 1

        def l1_loss(real, prediction):
            return tf.reduce_mean(tf.abs(real - prediction), axis=-1)

        self.loss_functions = [
            # tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0,name='cat_crossentropy',reduction=tf.keras.losses.Reduction.AUTO),
            tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, name='cat_crossentropy',
                                               reduction='none'),
            l1_loss, tf.losses.MSE]

        self.z1_mean = tf.keras.layers.Dense(self.end_z_dim, activation=None, use_bias=False,
                                             input_shape=(-1, self.end_z_dim), name='mean', dtype=dtype)

        self.z1_log_var = tf.keras.layers.Dense(end_z_dim, activation=None, use_bias=False,
                                                input_shape=(-1, self.end_z_dim), name='var', dtype=dtype)

        if self.NLL_lambda:
            end_dim = 2
        elif include_parallel_info:
            end_dim = 10
        else:
            end_dim = 1
        # small network:
        self.decoder = decoder_mlp_interpretable(batchnorm_axis, end_z_dim, [20, 10, 5], activation,
                                                 batchnorm_renorm, end_dim=end_dim)

        if self.architecture == 'VAE':
            print('VAE')


        else:
            print('AE')
        if exclude_sf:
            print('exclude SF')
            self.exclude_sf_loss = tf.losses.MSE
            if enc_type == 'sim':
                input_dim = 16
            else:
                input_dim = end_z_dim
            self.exclude_sf_clf = decoder_mlp_interpretable(batchnorm_axis, input_dim, layer_dims, activation,
                                                            batchnorm_renorm, end_dim=1, name='sf_clf')
            # self.exclude_sf_opt= tf.keras.optimizers.Adam(learning_rate=learning_rate)#learning_rate# 0.001
            if opt == 'adam':
                self.exclude_sf_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                # self.exclude_sf_opt =tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            elif opt == 'sgd':
                self.exclude_sf_opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        if opt == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        elif opt == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif opt == 'rms':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)



    def train_iterations_lstm(self,dataset, num_train_generations,beta=1):
        meta_data=np.zeros(7)
        if self.exclude_sf:
            num_losses=3 + len(self.loss_functions)+1
        else:
            num_losses=3 + len(self.loss_functions)
        loss_arr = np.zeros((num_losses))
        counter = 0
        counter2= 0
        counter_meta=0
        z_reps_mean=np.zeros(self.end_z_dim)
        z_reps_std=np.zeros(self.end_z_dim)
        #print('train')

        for (focus_snp,neighbors,distances) in dataset:
            if self.include_parallel_info:
                #print('train parallel',flush=True)
                loss_ = self.train_parallel(focus_snp,neighbors,distances,num_train_generations,beta=beta)
            else:
                #print('train normal',flush=True)
                loss_,z_reps_means,z_reps_stds,counter_tmp,meta_data_,count = self.train_lstm(focus_snp,neighbors,distances,num_train_generations,beta=beta)
                counter2+=counter_tmp
                z_reps_mean+=z_reps_means
                z_reps_std+=z_reps_stds
                meta_data+=meta_data_
                if count:
                    counter_meta+=1

            loss_arr += loss_

            counter += 1

        if counter_meta:
            meta_data=meta_data/counter_meta
        else:
            meta_data=meta_data
        return loss_arr / counter,z_reps_mean/counter2,z_reps_std/counter2,meta_data




    def train_lstm(self,focus_snp,neighbors,distances,num_train_generations,beta=0):
        meta_data=np.zeros(7)
        if self.exclude_sf:
            num_losses=3 + len(self.loss_functions)+1
        else:
            num_losses=3 + len(self.loss_functions)
        loss_arr = np.zeros((num_losses))
        z_reps_mean=np.zeros(self.end_z_dim)
        z_reps_std=np.zeros(self.end_z_dim)

        loss_counter=0
        meta_data_counter=0
        for gen_counter in range(num_train_generations-self.time_batch_size):
            loss_counter+=1
            if self.enc_type=='MLP' or 'sim' in self.enc_type:
                neighbor_data=neighbors[:,:,gen_counter:gen_counter+self.time_batch_size,0]
            else:
                neighbor_data=neighbors[:,:,gen_counter:gen_counter+self.time_batch_size]
            snp_output = focus_snp[:,gen_counter+self.time_batch_size]
            focus_snp_input=focus_snp[:,gen_counter:gen_counter+self.time_batch_size]
            main_losses, other_losses,z_reps_means,z_reps_stds,attention,grads,pred_change = self.train_lstm_step(neighbor_data, focus_snp_input,snp_output,distances, beta=beta)
            loss_arr[0] += main_losses[0]
            loss_arr[1] += main_losses[1]
            loss_arr[2] += main_losses[2]
            if self.debug:
                afc_neighbors=tf.abs(neighbors[:,:,0,0]-neighbors[:,:,-1,0])
                focus_snp_input_afc=tf.abs(tf.expand_dims(focus_snp_input[:,0]-focus_snp_input[:,-1],axis=1))
                direction_similarity=afc_neighbors-focus_snp_input_afc
                tmp_afc_max=tf.maximum(afc_neighbors,
                                        tf.broadcast_to( focus_snp_input_afc, tf.shape(afc_neighbors)))


                tmp1,tmp2=distances.numpy().flatten(), attention.numpy().flatten()
                tmp3=np.abs(direction_similarity.numpy().flatten())
                tmp_afc_max=tmp_afc_max.numpy().flatten()
                plt_afc, plt_linkage, plt_attention = [], [], []
                for (afc_max,link,att,afc) in zip(tmp_afc_max,tmp1,tmp2,tmp3):
                    if afc_max > 0.01:
                        plt_afc.append(afc)
                        plt_linkage.append(link)
                        plt_attention.append(att)


                if len(plt_afc):
                    res = stats.spearmanr(tmp1,tmp2)
                    meta_data[0]+=np.mean(res.statistic)#

                    res = stats.spearmanr(tmp1, tmp3)
                    flatted_grads=[]
                    for elem in grads:
                        if elem is not None:
                            flatted_grads.extend(elem.numpy().flatten())
                    flatted_grads=np.asarray(flatted_grads)

                    meta_data[1]+=res.statistic
                    meta_data[2]+=np.min(flatted_grads)
                    meta_data[3] += np.mean(flatted_grads)
                    meta_data[4] += np.max(flatted_grads)
                    meta_data[5]+=np.mean(pred_change)
                    meta_data[6]+=np.mean(snp_output-focus_snp_input[:,-1])
                    meta_data_counter+=1
            if self.exclude_sf:
                loss_arr[-1]+=main_losses[3]


            z_reps_mean += z_reps_means.numpy()
            z_reps_std += z_reps_stds.numpy()

            for i in range(len(self.loss_functions)):
                loss_arr[3 + i] += tf.reduce_mean(other_losses[i]).numpy()
        if meta_data_counter:
            meta_data_return=meta_data/meta_data_counter
        else:
            meta_data_return=meta_data
        return loss_arr/loss_counter,z_reps_mean,z_reps_std,loss_counter,meta_data_return,meta_data_counter



    def train_parallel(self,focus_snp,neighbors,distances,num_train_generations,beta=0):
        loss_arr = np.zeros((3 + len(self.loss_functions)))
        loss_counter=0
        for gen_counter in range(num_train_generations-self.time_batch_size-1):
            loss_counter+=1
            neighbor_data=neighbors[:,:,:,gen_counter:gen_counter+self.time_batch_size]
            #snp_input=all_snp_batch[:,gen_counter:gen_counter+self.time_batch_size]
            snp_output = focus_snp[:,:,gen_counter+self.time_batch_size]#all_snp_batch[:,gen_counter+self.time_batch_size]
            focus_snp_input=focus_snp[:,:,gen_counter:gen_counter+self.time_batch_size]
            loss, l1_loss, kl_loss, other_losses = self.train_parallel_step(neighbor_data, focus_snp_input,snp_output,distances, beta=beta)
            loss_arr[0] += loss
            loss_arr[1] += l1_loss
            loss_arr[2] += kl_loss
            for i in range(len(self.loss_functions)):
                loss_arr[3 + i] += tf.reduce_mean(other_losses[i]).numpy()
        return loss_arr/loss_counter

    @tf.function
    def train_lstm_step(self,neighbor_data, focus_snp_input,snp_output,distances,beta=0):

        if self.exclude_sf:
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                #
                t1, loss, l1_loss, kl_loss,  freq_change, all_reconstruction_losses,z_reps_means,z_reps_stds,attention,_= self.vae_execution(
                    neighbor_data, focus_snp_input,snp_output,distances,
                    beta=beta,
                    training=True)
                if self.model_part == 'SF':
                    neg_sf=-loss[1]

            train_vars = self.trainable_variables
            train_vars1, train_vars2 = [], []
            for var in train_vars:
                if 'sf_clf' in var.name:
                    train_vars2.append(var)
                else:
                    train_vars1.append(var)

            if self.model_part == '':

                grads = tape1.gradient(loss[0], train_vars1)
                self.optimizer.apply_gradients(zip(grads, train_vars1))

                grads2 = tape2.gradient(loss[1], train_vars2)
                self.exclude_sf_opt.apply_gradients(zip(grads2, train_vars2))
            elif self.model_part == 'F':
                # train encoder + decoder:
                grads = tape1.gradient(loss[0], train_vars1)
                self.optimizer.apply_gradients(zip(grads, train_vars1))

                grads2 = tape2.gradient(loss[1], train_vars2)
                self.exclude_sf_opt.apply_gradients(zip(grads2, train_vars2))
            elif self.model_part == 'SF':
                # train encoder
                train_vars1_names=[var.name for var in train_vars1]
                train_decoder_names=[var.name for var in self.decoder.trainable_variables]
                enc_var_names=set(train_vars1_names).difference(set(train_decoder_names))
                enc_vars=[var for var in train_vars1 if var.name in enc_var_names]

                grads2 = tape2.gradient(loss[1], train_vars2)
                self.exclude_sf_opt.apply_gradients(zip(grads2, train_vars2))
                #print(neg_sf)
                grads = tape1.gradient(neg_sf, enc_vars)
                self.optimizer.apply_gradients(zip(grads, enc_vars))



            return [loss[0], l1_loss, kl_loss,loss[1]], all_reconstruction_losses,z_reps_means,z_reps_stds,attention,0,0
        else:
            with tf.GradientTape() as tape:
                #
                t1, loss, l1_loss, kl_loss,  freq_change, all_reconstruction_losses,z_reps_means,z_reps_stds,attention,pred_change= self.vae_execution(
                    neighbor_data, focus_snp_input,snp_output,distances,
                    beta=beta,
                    training=True)
                loss = tf.reduce_mean(loss)
            train_vars = self.trainable_variables
            grads = tape.gradient(loss, train_vars)
            #print('DEBUG net data:',loss,
            #      'count nans in gradient',sum([ np.count_nonzero(np.isnan(entry.numpy()))for entry in grads]),
            #      'example focus snp',focus_snp_input.numpy()[:2],flush=True)
            self.optimizer.apply_gradients(zip(grads, train_vars))

            return [loss, l1_loss, kl_loss], all_reconstruction_losses,z_reps_means,z_reps_stds,attention,grads,pred_change

    @tf.function
    def train_parallel_step(self,neighbor_data, focus_snp_input,snp_output,distances,beta=0):
        with tf.GradientTape() as tape:
            #
            t1, loss, l1_loss, kl_loss,  freq_change, all_reconstruction_losses= self.vae_execution_parallel(
                neighbor_data, focus_snp_input,snp_output,distances,
                beta=beta,
                training=True)
            loss = tf.reduce_mean(loss)
        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return l1_loss, kl_loss, loss, all_reconstruction_losses


    def gene_encoder_exec(self,elems):
        gene_encoding= self.encoder_gene(elems, training=True)
        return gene_encoding
    def snp_encoder_exec(self,elems):
        snp_encoding= self.encoder_snp_window(elems, training=True)
        return snp_encoding
    def snp_encoder_exec_parallel(self,elems):
        snp_encoding= self.encoder_focus_snp(elems, training=True)#self.encoder_snp_window(elems, training=True)
        return snp_encoding
    def snp_encoder_exec_parallel_eval(self,elems):
        snp_encoding= self.encoder_focus_snp(elems, training=False)#self.encoder_snp_window(elems, training=True)
        return snp_encoding
    def snp_gene_combi(self,input_tuple):
        return self.snp_gene_combiner(self.gene_concat(input_tuple))


    def linkage_weights(self,snp_encoding_window):
        #dim: #batch_size, time, enc_dim
        deltas = snp_encoding_window[:,1:] - snp_encoding_window[:,:-1]
        #abs_mean = tf.reduce_sum(tf.math.reduce_mean(tf.abs(deltas), axis=[2], keepdims=True),axis=0,keepdims=True)
        abs_variance =  tf.reduce_sum(tf.math.reduce_std(tf.abs(deltas), axis=2),axis=1, keepdims=True)#, keepdims=True
        multiplied_var = abs_variance * self.var_multiplier
        var_exp = tf.exp(-multiplied_var)
        return var_exp
    @tf.function
    def get_attention(self, neighbor_data, focus_snp_input):
        attention,softmax_att=0,0
        if self.integrate_gene and 'sim' in self.enc_type:
            focus_snp_time_series, attention, focus_enc1,softmax_att = self.encoder_snp_window(focus_snp_input, neighbor_data)
        return attention,softmax_att

    @tf.function
    def vae_execution(self, neighbor_data, focus_snp_input, snp_output, distances, beta=0, training=True):
        parallel_iterations = self.time_batch_size

        attention = 0
        if self.integrate_gene:
            if self.enc_type == 'MLP':
                focus_snp_time_series = self.encoder_focus_snp(focus_snp_input)
                snp_input = tf.transpose(neighbor_data, (2, 0, 1))
                snp_encoding_window = tf.map_fn(fn=self.snp_encoder_exec, elems=snp_input, dtype=dtype,
                                                parallel_iterations=parallel_iterations)
                snp_encoding_window = tf.stack(snp_encoding_window)
                snp_encoding_window = tf.transpose(snp_encoding_window, (1, 0, 2))
                snp_encoding_window = self.time_combiner(
                    tf.reshape(snp_encoding_window, (-1, self.time_batch_size * self.end_z_dim)))
                focus_snp_time_series = self.z1(self.concadenate([focus_snp_time_series, snp_encoding_window]))
            elif 'sim' in self.enc_type:
                focus_snp_time_series, attention, focus_enc1, _ = self.encoder_snp_window(focus_snp_input,
                                                                                          neighbor_data)
            else:
                focus_snp_time_series = self.encoder_focus_snp(focus_snp_input)
                snp_encoding_window = self.encoder_snp_window(neighbor_data)
                focus_snp_time_series = self.z1(self.concadenate([focus_snp_time_series, snp_encoding_window]))
        else:
            focus_snp_time_series = self.encoder_focus_snp(focus_snp_input)

        snp_encoding_gauss = focus_snp_time_series

        # vae stuff:
        if self.architecture == 'VAE':
            cls_input0, cls_input_var, cls_input_mean = self.gauss(snp_encoding_gauss, training=training)
        else:
            cls_input0 = self.z1_mean(snp_encoding_gauss)

        t1 = self.decoder(cls_input0, training=training)
        t2 = tf.expand_dims(snp_output, axis=1)
        if self.multiply_result:
            f1 = tf.expand_dims(focus_snp_input[:, -1], axis=1)
            t1 = 1 / (1 + ((1 - f1) / (tf.exp(t1) * f1)))
            pred_change = t1 - focus_snp_input[:, -1]
        else:
            t1 = tf.keras.activations.sigmoid(t1)
            pred_change = t1 - focus_snp_input[:, -1]

        if self.architecture == 'VAE':
            # KL divergence term (per latent dim)
            kl_per_dim = -0.5 * (1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var))
            # free bits: clamp minimum KL per dim
            kl_per_dim = tf.maximum(kl_per_dim, self.free_bits)
            kl_loss = tf.reduce_sum(kl_per_dim, axis=1)
            kl_loss = tf.reduce_mean(kl_loss)

            # kl_loss = -0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var),axis=1)
        else:
            kl_loss = 0
        if self.exclude_sf:
            # l1_loss = self.loss_functions[self.loss_idx](t2, tf.keras.activations.sigmoid(t1*focus_snp_input[:,0:1]))
            l1_loss = self.loss_functions[self.loss_idx](t2, t1)  # *10
            if self.enc_type == 'sim':
                sf = self.exclude_sf_clf(focus_enc1, training=training)
            else:
                sf = self.exclude_sf_clf(cls_input0, training=training)

            sf_loss = self.exclude_sf_loss(focus_snp_input[:, 0:1], sf)
            if self.model_part == '':
                loss = l1_loss - (sf_loss * self.sf_ex_alpha) + (beta * kl_loss)
            else:
                loss = l1_loss + (beta * kl_loss)
            loss = tf.reduce_mean(loss)
            sf_loss = tf.reduce_mean(sf_loss)
            loss = (loss, sf_loss)
        else:
            l1_loss = self.loss_functions[self.loss_idx](t2, t1)  # *10
            l1_loss = tf.reduce_mean(l1_loss)

            loss = l1_loss + (beta * kl_loss)

        all_reconstruction_losses = [loss_f(t2, t1) for loss_f in self.loss_functions]
        freq_change = 0
        z_reps_means = tf.reduce_mean(cls_input_mean, axis=0)
        z_reps_stds = tf.reduce_mean(tf.exp(cls_input_var), axis=0)
        return t1, loss, l1_loss, kl_loss, freq_change, all_reconstruction_losses, z_reps_means, z_reps_stds, attention, pred_change

    def vae_execution_parallel(self,neighbor_data, focus_snp_input,snp_output,distances,beta=0,training=True):
        parallel_iterations = self.time_batch_size

        #plot_vars = dict()
        #'''
        if self.recursive_enc and self.integrate_gene:
            snp_input = tf.transpose(neighbor_data, (2, 0, 1))
            if self.use_distance_weights:
                snp_input=snp_input*distances

            snp_encoding_window = tf.map_fn(fn=self.snp_encoder_exec, elems=snp_input, dtype=dtype,
                                            parallel_iterations=parallel_iterations)
            #snp_encoding_window = tf.stack(snp_encoding_window)
            #snp_encoding_window = tf.transpose(snp_encoding_window, (1, 0, 2))
            snp_encoding_window = self.encoder_LSTM(snp_encoding_window)

            focus_snp_input=tf.expand_dims(focus_snp_input,axis=-1)
            focus_snp_time_series = self.z1(focus_snp_input)


            combine=tf.concat([focus_snp_time_series,snp_encoding_window],axis=-1)
            snp_encoding_gauss = self.combiner(combine)


        else:
            focus_snp_input = tf.transpose(focus_snp_input, (1, 0, 2))
            focus_snp_time_series= tf.map_fn(fn=self.snp_encoder_exec_parallel, elems=focus_snp_input, dtype=dtype,
                      parallel_iterations=parallel_iterations)

            focus_snp_time_series= tf.transpose(focus_snp_time_series, (1, 0,2))
            focus_snp_time_series=tf.reshape(focus_snp_time_series,(-1,10*self.end_z_dim))
            focus_snp_time_series=self.rep_combiner(focus_snp_time_series)
            #focus_snp_time_series = self.encoder_focus_snp(focus_snp_input)
            if self.integrate_gene:

                snp_input = tf.transpose(neighbor_data, (0,1, 3, 2))
                if self.use_distance_weights:
                    snp_input = snp_input * distances

                snp_encoding_window = self.encoder_snp_window(snp_input)


                focus_snp_time_series=self.z1(self.concadenate([focus_snp_time_series,snp_encoding_window]))

            snp_encoding_gauss=focus_snp_time_series


        # vae stuff:
        if self.architecture == 'VAE':
            num_reps=snp_input.shape[-1]
            cls_input_mean = self.z1_mean(snp_encoding_gauss, training=training)
            cls_input_var = self.z1_log_var(snp_encoding_gauss, training=training)
            exp0 = tf.exp(cls_input_var * .5)

            cls_input_mean_ = tf.tile(tf.expand_dims(cls_input_mean,axis=0), [num_reps, 1, 1])
            exp0_ = tf.tile(tf.expand_dims(exp0, axis=0), [num_reps, 1, 1])

            eps0 = tf.random.normal(shape=(num_reps,cls_input_mean.shape[0],cls_input_mean.shape[1]), mean=0, stddev=1, dtype=dtype)
            cls_input0 = eps0 * exp0_ + cls_input_mean_

            dec_output= tf.map_fn(fn=lambda x:self.decoder(x,training=training), elems=cls_input0, dtype=dtype,parallel_iterations=parallel_iterations)
            dec_output=tf.squeeze(dec_output)
            t1=tf.transpose(dec_output,(1,0))

        else:
            cls_input0 = self.z1_mean(snp_encoding_gauss)

            t1 = self.decoder(cls_input0,training=training)

        t2 = snp_output

        if self.loss_type=='EMD':
            l1_loss = self.loss_functions[2](t2, t1)
        elif self.loss_type=='KDE':
            l1_loss = kde_loss(snp_output, t1, 0.1)
        else:
            l1_loss =self.loss_functions[2](t2, t1)

        #l1_loss_tmp = emd_loss(snp_output, t1)
        #self.loss_functions[self.loss_idx](t2, t1)

        if self.architecture == 'VAE':
            kl_loss = -0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var),axis=1)
        else:
            kl_loss = 0

        loss = l1_loss + (beta * kl_loss)
        l1_loss = tf.reduce_mean(l1_loss)
        kl_loss = tf.reduce_mean(kl_loss)
        loss = tf.reduce_mean(loss)
        all_reconstruction_losses = [tf.reduce_mean(loss_f(t2, t1)) for loss_f in self.loss_functions]
        freq_change=0
        return t1, loss, l1_loss, kl_loss,  freq_change, all_reconstruction_losses#[:2]


    def gauss(self,input_lay,training=False):
        cls_input_mean0 = self.z1_mean(input_lay,training=training)
        cls_input_var0 = self.z1_log_var(input_lay,training=training)

        eps0 = tf.random.normal(shape=cls_input_mean0.shape, mean=0, stddev=1, dtype=dtype)
        exp0 = tf.exp(cls_input_var0 * .5)
        cls_input0 = eps0 * exp0 + cls_input_mean0
        return cls_input0,cls_input_var0,cls_input_mean0


    def snp_encoder_exec_test(self,elems):
        snp_encoding= self.encoder_snp_window(elems, training=False)
        return snp_encoding



    @tf.function
    def eval(self,snp_input,neighbor_data,distances):

        parallel_iterations = self.time_batch_size


        if self.integrate_gene:
            if self.enc_type == 'MLP':
                focus_snp_time_series = self.encoder_focus_snp(snp_input)
                snp_encoding_window = tf.map_fn(fn=self.snp_encoder_exec_test, elems=neighbor_data, dtype=dtype,
                                      parallel_iterations=parallel_iterations)
                snp_encoding_window=tf.stack(snp_encoding_window)
                snp_encoding_window=tf.transpose(snp_encoding_window,(1,0,2))
                snp_encoding_window_combi=self.time_combiner(tf.reshape(snp_encoding_window,(-1,self.time_batch_size*self.end_z_dim)))
                focus_snp_time_series = self.z1(self.concadenate([focus_snp_time_series, snp_encoding_window_combi]))
            elif self.enc_type=='sim':
                neighbor_data=tf.transpose(neighbor_data,(1,2,0))
                focus_snp_time_series,snp_encoding_window_combi,focus_enc1,_ = self.encoder_snp_window(snp_input, neighbor_data)

            else:
                focus_snp_time_series = self.encoder_focus_snp(snp_input)
                neighbor_data=tf.transpose(neighbor_data,(1,2,0))
                snp_encoding_window_combi = self.encoder_snp_window(np.expand_dims(neighbor_data,axis=-1))
                focus_snp_time_series = self.z1(self.concadenate([focus_snp_time_series, snp_encoding_window_combi]))
        else:
            snp_encoding_window_combi=None
            focus_snp_time_series = self.encoder_focus_snp(snp_input)
        snp_encoding_gauss = focus_snp_time_series

        # vae stuff:
        if self.architecture == 'VAE':
            cls_input0, cls_input_var, cls_input_mean = self.gauss(snp_encoding_gauss,training=False)
        else:
            cls_input0 = self.z1_mean(snp_encoding_gauss)

        t1 = self.decoder(cls_input0,training=False)
        if self.multiply_result:
            f1=tf.cast(tf.expand_dims(snp_input[:,-1],axis=1),tf.float32)
            t1=1/(1+((1-f1)/(tf.exp(t1)*f1)))
        else:
            t1=tf.keras.activations.sigmoid(t1)


        return t1, cls_input0, snp_encoding_gauss, snp_encoding_window_combi


    def eval_parallel(self, snp_input, neighbor_data, distances,training=False):
        #TODO
        parallel_iterations = self.time_batch_size

        if self.recursive_enc and self.integrate_gene:
            print('TODO')
        else:
            #focus_snp_input = tf.transpose(snp_input, (1, 0, 2))
            focus_snp_time_series = tf.map_fn(fn=self.snp_encoder_exec_parallel_eval, elems=snp_input, dtype=dtype,
                                              parallel_iterations=parallel_iterations)

            focus_snp_time_series = tf.transpose(focus_snp_time_series, (1, 0, 2))
            focus_snp_time_series = tf.reshape(focus_snp_time_series, (-1, 10 * self.end_z_dim))
            focus_snp_time_series = self.rep_combiner(focus_snp_time_series)
            # focus_snp_time_series = self.encoder_focus_snp(focus_snp_input)
            if self.integrate_gene:

                #snp_input = tf.transpose(neighbor_data, (0, 1, 3, 2))
                if self.use_distance_weights:
                    neighbor_data = neighbor_data * distances

                snp_encoding_window = self.encoder_snp_window(neighbor_data)

                focus_snp_time_series = self.z1(self.concadenate([focus_snp_time_series, snp_encoding_window]))

            snp_encoding_gauss = focus_snp_time_series

        # vae stuff:
        if self.architecture == 'VAE':
            num_reps=neighbor_data.shape[-1]
            cls_input_mean = self.z1_mean(snp_encoding_gauss, training=training)
            cls_input_var = self.z1_log_var(snp_encoding_gauss, training=training)
            exp0 = tf.exp(cls_input_var * .5)

            cls_input_mean_ = tf.tile(tf.expand_dims(cls_input_mean,axis=0), [num_reps, 1, 1])
            exp0_ = tf.tile(tf.expand_dims(exp0, axis=0), [num_reps, 1, 1])

            eps0 = tf.random.normal(shape=(num_reps,cls_input_mean.shape[0],cls_input_mean.shape[1]), mean=0, stddev=1, dtype=dtype)
            cls_input0 = eps0 * exp0_ + cls_input_mean_

            dec_output= tf.map_fn(fn=lambda x:self.decoder(x,training=training), elems=cls_input0, dtype=dtype,parallel_iterations=parallel_iterations)
            dec_output=tf.squeeze(dec_output)
            t1=tf.transpose(dec_output,(1,0))

        else:
            cls_input0 = self.z1_mean(snp_encoding_gauss)

            t1 = self.decoder(cls_input0,training=training)
        #return t1.numpy(), cls_input0[0].numpy(), snp_encoding_gauss.numpy(),snp_encoding_window.numpy()
        return t1, cls_input0, snp_encoding_gauss, snp_encoding_window


def kernel_density_estimation(input):
    x, data_points, bandwidth=input
    # Compute kernel density estimation using Gaussian kernel
    kernel_values = tf.exp(-0.5 * ((x - data_points) / bandwidth) ** 2)
    density_estimation = tf.reduce_mean(kernel_values, axis=-1)
    return density_estimation


def kde_loss(y_true, y_pred, bandwidth,num_parallel_runs=10):
    # Compute KDE-based negative log likelihood loss
    # TODO :run this in parallel
    y_pred=tf.transpose(y_pred,(1,0))
    y_pred=tf.expand_dims(y_pred,axis=-1)
    y_pred=tf.repeat(y_pred,num_parallel_runs,axis=-1)
    y_true = tf.expand_dims(y_true,axis=0)
    y_true=tf.repeat(y_true,num_parallel_runs,axis=0)
    bandwidth=tf.repeat(bandwidth,num_parallel_runs,axis=0)
    predicted_density = tf.map_fn(fn=kernel_density_estimation, elems=(y_pred,y_true,bandwidth), dtype=dtype,
                                      parallel_iterations=num_parallel_runs)

    #predicted_density = kernel_density_estimation(y_pred, y_true, bandwidth)
    negative_log_likelihood = -tf.reduce_mean(tf.math.log(predicted_density + 1e-10),axis=0)
    return negative_log_likelihood





