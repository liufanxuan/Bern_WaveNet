import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, Add, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam
from keras import layers
from wavetrans import DWT_3D_Pooling,DWT_3D_Pooling_sa,IWT_3D_UpSampling

# from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

K.set_image_data_format("channels_first")
IMAGE_ORDERING = 'channels_first'
init = tf.keras.initializers.glorot_uniform()
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate
    
    
def unet_model_3d(input_shape, final_act=None, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[0]
    current_layer = inputs
    levels = list()
    n_t = 1
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization) 
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = DWT_3D_Pooling()(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    if final_act:
        act = Activation(final_act)(final_convolution)
        model = Model(inputs=inputs, outputs=act)
    else:
        model = Model(inputs=inputs, outputs=final_convolution)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]

    # if include_label_wise_dice_coefficients and n_labels > 1:
    #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    #     if metrics:
    #         metrics = metrics + label_wise_dice_metrics
    #     else:
    #         metrics = label_wise_dice_metrics

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model
    

    
def unet_se_3d(input_shape, final_act=None, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[0]
    current_layer = inputs
    levels = list()
    n_t = 1
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization) 
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = DWT_3D_Pooling()(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        current_layer = SE_block(current_layer,levels[layer_depth][1]._keras_shape[1], ratio = 4, name='att'+str(layer_depth))
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    if final_act:
        act = Activation(final_act)(final_convolution)
        model = Model(inputs=inputs, outputs=act)
    else:
        model = Model(inputs=inputs, outputs=final_convolution)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]

    # if include_label_wise_dice_coefficients and n_labels > 1:
    #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    #     if metrics:
    #         metrics = metrics + label_wise_dice_metrics
    #     else:
    #         metrics = label_wise_dice_metrics

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model
    
    
    
def unet_sase_3d(input_shape, final_act=None, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[0]
    current_layer = inputs
    levels = list()
    n_t = 1
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization) 
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = DWT_3D_Pooling_sa()(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        current_layer = SE_block(current_layer,levels[layer_depth][1]._keras_shape[1], ratio = 4, name='att'+str(layer_depth))
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    if final_act:
        act = Activation(final_act)(final_convolution)
        model = Model(inputs=inputs, outputs=act)
    else:
        model = Model(inputs=inputs, outputs=final_convolution)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]

    # if include_label_wise_dice_coefficients and n_labels > 1:
    #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    #     if metrics:
    #         metrics = metrics + label_wise_dice_metrics
    #     else:
    #         metrics = label_wise_dice_metrics

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model
    
def unet_sa_3d(input_shape, final_act=None, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[0]
    current_layer = inputs
    levels = list()
    n_t = 1
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization) 
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = DWT_3D_Pooling_sa()(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
       
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    if final_act:
        act = Activation(final_act)(final_convolution)
        model = Model(inputs=inputs, outputs=act)
    else:
        model = Model(inputs=inputs, outputs=final_convolution)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]

    # if include_label_wise_dice_coefficients and n_labels > 1:
    #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    #     if metrics:
    #         metrics = metrics + label_wise_dice_metrics
    #     else:
    #         metrics = label_wise_dice_metrics

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model
    
def unet_res_3d(input_shape, final_act=None, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[0]
    current_layer = inputs
    levels = list()
    n_t = 1
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization) 
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth == 0:
            layer2 = concatenate([current_layer,layer2], axis=1)
        if layer_depth < depth - 1:
            current_layer = DWT_3D_Pooling()(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    if final_act:
        act = Activation(final_act)(final_convolution)
        model = Model(inputs=inputs, outputs=act)
    else:
        model = Model(inputs=inputs, outputs=final_convolution)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]

    # if include_label_wise_dice_coefficients and n_labels > 1:
    #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    #     if metrics:
    #         metrics = metrics + label_wise_dice_metrics
    #     else:
    #         metrics = label_wise_dice_metrics

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model
        
def SE_block(x, out_dim, ratio, name, batch_norm=False):
    """
    self attention squeeze-excitation block, attention mechanism on channel dimension
    :param x: input feature map
    :return: attention weighted on channel dimension feature map
    """
    # Squeeze: global average pooling
    print(x)
    x_s = layers.GlobalAveragePooling3D(data_format=None)(x)
    print(x_s)
    # Excitation: bottom-up top-down FCs
    if batch_norm:
        x_s = layers.BatchNormalization()(x_s)
    x_e = layers.Dense(units=out_dim//ratio)(x_s)
    x_e = layers.Activation('relu')(x_e)
    if batch_norm:
        x_e = layers.BatchNormalization()(x_e)
    x_e = layers.Dense(units=out_dim)(x_e)
    print(x_e)
    x_e = layers.Activation('sigmoid')(x_e)
    x_e = layers.Reshape((out_dim, 1, 1, 1), name=name+'channel_weight')(x_e)
    result = layers.multiply([x, x_e])
    return result
    
     

def SE_block_2(x, out_dim, ratio, name, batch_norm=False):
    """
    self attention squeeze-excitation block, attention mechanism on channel dimension
    :param x: input feature map
    :return: attention weighted on channel dimension feature map
    """
    # Squeeze: global average pooling
    x_s = layers.GlobalAveragePooling3D(data_format=None)(x)
    # Excitation: bottom-up top-down FCs
    if batch_norm:
        x_s = layers.BatchNormalization()(x_s)
    x_e = layers.Dense(units=out_dim//ratio)(x_s)
    x_e = layers.Activation('relu')(x_e)
    if batch_norm:
        x_e = layers.BatchNormalization()(x_e)
    x_e = layers.Dense(units=out_dim)(x_e)
    x_e = layers.Activation('sigmoid')(x_e)
    x_e = layers.Reshape((out_dim, 1, 1, 1), name=name+'channel_weight')(x_e)
    result = layers.multiply([x, x_e])
    return result
def spatial_attention(inputs):

    # 在通道维度上做最大池化和平均池化[b,h,w,c]==>[b,h,w,1]
    # keepdims=Fale那么[b,h,w,c]==>[b,h,w]
    x_1 = create_convolution_block(inputs, n_filters=1, batch_normalization=False)  # 在通道维度求最大值
    x_2 = create_convolution_block(x_1, n_filters=1, batch_normalization=False)
    x_3 = create_convolution_block(x_2, n_filters=1, batch_normalization=False)

    # 在通道维度上堆叠[b,h,w,2]
    x = layers.concatenate([x_1, x_2, x_3])

    # 1*1卷积调整通道[b,h,w,1]
    x = layers.Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(x)

    # sigmoid函数权重归一化
    x = tf.nn.sigmoid(x)

    # 输入特征图和权重相乘
    x = layers.Multiply()([inputs, x])

    return x         
def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)