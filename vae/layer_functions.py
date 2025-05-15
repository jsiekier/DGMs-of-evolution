import tensorflow as tf
tf.keras.backend.set_floatx('float32')

dtype=tf.float32

def create_encoder_layers2D(filter_size, strides, dilations, num_filter, old_num_filter, use_shortcut, old_h, batch_ax,
                          batch_ren, var_name,trainable=True):
    conv = tf.keras.layers.Conv2D(num_filter, filter_size, dilation_rate=dilations, strides=strides, padding="valid",
                                  data_format='channels_last', use_bias=False, activation=None, name=var_name,trainable=trainable,dtype=dtype)
    # dropout=tf.keras.layers.Dropout(rate=)
    batch_norm = tf.keras.layers.BatchNormalization(axis=batch_ax, renorm=batch_ren,trainable=trainable,dtype=dtype)
    if use_shortcut:
        s_pool = tf.keras.layers.AveragePooling2D(pool_size=filter_size, strides=strides, padding="valid",
                                                  data_format="channels_last",trainable=trainable)
        s_dense = tf.keras.layers.Dense(1, activation=None, use_bias=True, input_shape=(-1, old_h, old_num_filter),trainable=trainable,dtype=dtype)

        return s_pool, s_dense, conv, batch_norm
    else:
        return None, None, conv, batch_norm

def enc_layer2D(self, net, encoder_batch_input, s_pool,shortcut, conv, batch_norm,trainable=True):
    convolution = conv(net)
    convolution=tf.squeeze(convolution,axis=-2)
    if shortcut:
        shortcut = s_pool(net)

        #Reduce shortcut
        if self.reduce_shortcut:
            shortcut=tf.squeeze(shortcut,axis=-1)
            s_dense = tf.keras.layers.Dense(1, activation=None, use_bias=True, input_shape=(-1, shortcut.shape[1], 1),trainable=trainable)
            shortcut = s_dense(shortcut)
        convolution = self.concadenate([convolution, shortcut])
    encoder_batch_input.append(convolution)
    if self.batchnorm_axis:
        convolution = batch_norm(convolution)
    if self.dropout:
        dropout = tf.keras.layers.Dropout(rate=self.dropout)
        convolution = dropout(convolution)

    convolution = self.activation(convolution)
    return convolution, encoder_batch_input
def decoder_mlp_interpretable(batchnorm_axis , z_dim  ,layer_dims,activation,batchnorm_renorm,end_dim,name='decoder'):
    print('Create Decoder')
    inputs = tf.keras.Input(shape=(z_dim,),dtype=dtype)

    net =inputs
    input_dim =z_dim

    print(layer_dims,z_dim,batchnorm_axis, flush=True)
    for lay in range(len(layer_dims)):
        output_dim =layer_dims[lay]

        net =tf.keras.layers.Dense(output_dim, activation=activation, use_bias=True ,input_shape=(-1 ,input_dim),dtype=dtype,name=name+str(lay)+'dense')(net)
        if batchnorm_axis:
            net= tf.keras.layers.BatchNormalization(axis=batchnorm_axis,dtype=dtype,name=name+str(lay)+'bn')(net)#renorm=batchnorm_renorm
        input_dim =output_dim


    net = tf.keras.layers.Dense(end_dim, activation=None, use_bias=False, input_shape=(-1, input_dim),
                                dtype=dtype,name=name+str(len(layer_dims))+'dense')(net)

    model = tf.keras.Model(inputs=inputs, outputs=[net])
    return model

def make_focus_snp_encoder(self,  batchnorm_axis,  input_length, layers, encoder_name='focus_snp_enc'):
    inputs = tf.keras.Input(shape=(input_length), dtype=dtype)
    mlp_net = inputs

    old_h = layers[0]

    for layer, mlp_dim in enumerate(layers+[self.end_z_dim]):  # 150,100,
        mlp_net = tf.keras.layers.Dense(mlp_dim, activation=self.activation,
                                        use_bias=True, input_shape=(-1, old_h),
                                        name=encoder_name + str(layer), trainable=True, dtype=dtype
                                        )(mlp_net)
        if batchnorm_axis:
            mlp_net = tf.keras.layers.BatchNormalization(axis=batchnorm_axis#, renorm=self.batchnorm_renorm
                                                         , trainable=True, dtype=dtype)(mlp_net)

        old_h = mlp_dim

    model = tf.keras.Model(inputs=inputs, outputs=mlp_net)
    return model, old_h, self.end_z_dim


def create_encoder_layers(filter_size, strides, dilations, num_filter, old_num_filter, use_shortcut, old_h, batch_ax,
                          batch_ren, var_name,trainable=True):
    conv = tf.keras.layers.Conv1D(num_filter, filter_size, dilation_rate=dilations, strides=strides, padding="same",
                                  data_format='channels_last', use_bias=False, activation=None, name=var_name,trainable=trainable,dtype=dtype)
    # dropout=tf.keras.layers.Dropout(rate=)
    batch_norm = tf.keras.layers.BatchNormalization(axis=batch_ax, renorm=batch_ren,trainable=trainable,dtype=dtype)
    if use_shortcut:
        s_pool = tf.keras.layers.AveragePooling1D(pool_size=filter_size, strides=strides, padding="same",
                                                  data_format="channels_last",trainable=trainable)
        s_dense = tf.keras.layers.Dense(1, activation=None, use_bias=True, input_shape=(-1, old_h, old_num_filter),trainable=trainable,dtype=dtype)

        return s_pool, s_dense, conv, batch_norm
    else:
        return None, None, conv, batch_norm


def enc_layer(self, net, s_pool, s_dense, conv, batch_norm):
    convolution = conv(net)
    if s_dense != None:
        shortcut = s_pool(net)

        if self.reduce_shortcut:
            shortcut = s_dense(shortcut)
        convolution = self.concadenate([convolution, shortcut])

    if self.batchnorm_axis:
        convolution = batch_norm(convolution)
    if self.dropout:
        dropout = tf.keras.layers.Dropout(rate=self.dropout)
        convolution = dropout(convolution)

    convolution = self.activation(convolution)
    return convolution
def make_window_CNN(self,  batchnorm_axis,  input_length, encoder_name='focus_snp_enc_cnn',filter_size=6,use_resnet=1):
    strides = [2, 2, 1,1]
    inputs = tf.keras.Input(shape=input_length, dtype=dtype)
    net = inputs
    # do first 2D conv
    encoder_batch_input = []
    old_num_filters = input_length[-1]
    old_h = input_length[0]

    s_pool, s_dense, conv, batch_norm = create_encoder_layers2D((1,input_length[1]), 1, 1,
                                                              64, old_num_filters, 0, old_h,
                                                              batchnorm_axis, self.batchnorm_renorm,
                                                              var_name=encoder_name + '_conv2D_', trainable=True)
    net, encoder_batch_input = enc_layer2D(self, net, encoder_batch_input, s_pool, s_dense, conv, batch_norm,)
    old_num_filters = 64
    old_h = net.shape[1]

    for layer, stride in enumerate(strides):

        if layer in self.gauss_lay and self.gauss_std:
            net = tf.keras.layers.GaussianNoise(stddev=self.gauss_std)(net)
        num_filter = 32 * (2 ** layer)
        if num_filter > 256:
            num_filter = 256


        if stride % 2 == 0:
            dilat = 1
        else:
            dilat = self.dilations

        s_pool, s_dense, conv, batch_norm = create_encoder_layers(filter_size, stride, dilat,
                                                                  num_filter, old_num_filters, use_resnet, old_h,
                                                                  batchnorm_axis, self.batchnorm_renorm,
                                                                  var_name=encoder_name+'_conv_' + str(layer)+'-'+str(stride),trainable=True)
        net =enc_layer(self,net, s_pool, s_dense, conv, batch_norm)


        if use_resnet:
            if self.reduce_shortcut:
                old_num_filters=num_filter+1#+old_num_filters#
            else:
                old_num_filters=num_filter+old_num_filters
        else:
            old_num_filters=num_filter
        old_h = int(old_h / stride)


    net = tf.keras.layers.Flatten()(net)#tf.reduce_mean(net, axis=1)
    mlp_input_shape=net.shape[1]


    net=tf.keras.layers.Dense(self.end_z_dim, activation=self.activation, use_bias=True,input_shape=(-1, mlp_input_shape*old_num_filters),name='encoder_mlp')(net)



    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model






