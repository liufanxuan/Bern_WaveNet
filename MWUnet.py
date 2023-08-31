import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Add, UpSampling2D, Activation, BatchNormalization, PReLU, Deconvolution2D,Lambda
from keras.optimizers import Adam
from keras import layers
from wavetrans import DWT_Pooling, IWT_UpSampling, DWT_Pooling_Db4

# from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

K.set_image_data_format("channels_last")
 
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_mw_2d(input_shape, final_act=None, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=3, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[2]
    current_layer = inputs
    print(current_layer)
    levels = list()
    n_t = 1
    current_layer = IWT_UpSampling()(current_layer)
    print(current_layer._keras_shape)
    print(current_layer)
    layer1 = create_convolution_block(input_layer=current_layer, n_filters=32,
                                          batch_normalization=batch_normalization)
    layer2 = create_convolution_block(input_layer=layer1, n_filters=64,
                                          batch_normalization=batch_normalization)
    print(current_layer)
    current_layer = DWT_Pooling()(layer2)
    print(current_layer)
    levels.append([layer1, layer2, current_layer])
    # add levels with max pooling
    for layer_depth in range(depth):
        
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**(layer_depth+1)),batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**(layer_depth+1))*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth+1][1]], axis=3)
        current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
    up_convolution = IWT_UpSampling()(current_layer)
    concat = concatenate([up_convolution, levels[0][1]], axis=3)
    current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=concat, batch_normalization=batch_normalization)
    current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
    final_convolution = Conv2D(1, (1, 1))(current_layer)
    final_convolution = DWT_Pooling()(final_convolution)                                                 
                                                 
                                                 
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



def unet_mw_2d_2(input_shape, final_act=None, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[2]
    current_layer = inputs
    levels = list()
    n_t = 1

    # add levels with max pooling
    for layer_depth in range(depth):
        
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**(layer_depth)),batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**(layer_depth))*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = DWT_Pooling()(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = IWT_UpSampling()(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=3)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[3],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[3],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
 
    final_convolution = Conv2D(n_labels, (1, 1))(current_layer)
                                                
                                                 
                                                 
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

def unet_mw_2d_plus(input_shape, final_act=None, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=3, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = input_shape[2]
    current_layer = inputs
    levels = list()
    n_t = 1
    layer1 = create_convolution_block(input_layer=current_layer, n_filters=32,
                                          batch_normalization=batch_normalization)
    layer2 = create_convolution_block(input_layer=layer1, n_filters=64,
                                          batch_normalization=batch_normalization)
    layer3 = create_convolution_block(input_layer=layer2, n_filters=32,
                                          batch_normalization=batch_normalization)
    output1 = create_convolution_block(input_layer=layer3, n_filters=1,
                                          batch_normalization=batch_normalization)
    current_layer = DWT_Pooling()(output1)
    levels.append([layer1, output1, current_layer])
    # add levels with max pooling
    for layer_depth in range(depth):
        
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**(layer_depth+1)),batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**(layer_depth+1))*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[3])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth+1][1]], axis=3)
        current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
    output2 = Conv2D(4, (1, 1))(current_layer)
    output2 = IWT_UpSampling()(output2)
    concat = output2
    #concatenate([output2, levels[0][1]], axis=3)
    current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=concat, batch_normalization=batch_normalization)
    current_layer = create_convolution_block(n_filters=levels[layer_depth+1][1]._keras_shape[3],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
    
    output3 = Conv2D(1, (1, 1))(current_layer)  
    weight_1 = Lambda(lambda x:x*0.02)
    weight_2 = Lambda(lambda x:x*0.9)
    weight_3 = Lambda(lambda x:x*0.08)
    output1 = weight_1(output1)
    output2 = weight_2(output2)
    output3 = weight_3(output3)                                               
    final_convolution =  layers.Add()([output1,output2,output3])                                            
                                                 
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


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3), activation=None,
                             padding='same', strides=(1, 1), instance_normalization=False):
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
    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
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


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2), strides=(2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution2D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling2D(size=pool_size)
        
        
