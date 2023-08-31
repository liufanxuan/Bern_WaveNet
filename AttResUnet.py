import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, PReLU, Deconvolution2D
from keras.optimizers import Adam
from keras import layers

# from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def Att_Resunet_2d(input_shape, final_act=None, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    n_labels = 4
    current_layer = inputs
    levels = list()
    n_t = 1
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_t*n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_t*n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        shortcut = layers.Conv2D(n_t*n_base_filters*(2**layer_depth)*2, kernel_size=(1,1), padding='same')(current_layer)
        if batch_normalization == True:
            shortcut = layers.BatchNormalization(axis=1)(shortcut)
        layer2 = layers.add([shortcut, layer2])
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        gate_sig = gating_signal(current_layer, levels[layer_depth][1]._keras_shape[1], batch_normalization)
        atten_layer = attention_block(levels[layer_depth][1], gate_sig, levels[layer_depth][1]._keras_shape[1])
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, atten_layer], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
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
        
def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    print(shape_x)
    shape_g = K.int_shape(gating)
    print(shape_g)

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)
    print(shape_theta_x)

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    print(K.int_shape(phi_g))
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]),
                                 padding='same')(phi_g)  # 16
    print(K.int_shape(upsample_g))

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[1])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[1], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def expend_as(tensor, rep):
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1),
                          arguments={'repnum': rep})(tensor)
                          
