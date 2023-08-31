import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, PReLU, Deconvolution2D, ZeroPadding2D, Reshape, Softmax
from keras.optimizers import Adam
from keras import layers

K.set_image_data_format("channels_first")
IMAGE_ORDERING = 'channels_first'


try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def get_resnet50_encoder(shape, pretrained='imagenet',
                         include_top=True, weights='imagenet',
                         input_tensor=None, input_shape=None,
                         pooling=None,
                         classes=1000):

    img_input = Input(shape)

    bn_axis = 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    f1 = x
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')
    f2 = x

    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='e')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='f')
    f4 = x
    
    x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c')

    f5 = x


    return img_input, [f1, f2, f3, f4, f5]
def get_resnet4level_encoder(shape, pretrained='imagenet',
                         include_top=True, weights='imagenet',
                         input_tensor=None, input_shape=None,
                         pooling=None,
                         classes=1000):

    img_input = Input(shape)

    bn_axis = 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    f1 = x
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')
    f2 = x

    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='e')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='f')
    f3 = x

    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')

    f4 = x
    



    return img_input, [f1, f2, f3, f4]    
    
def unet_resnet_2d(input_shape, encoder = get_resnet50_encoder, final_act=None, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=5, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

     
    n_labels = 4
    
    n_t = 1

    inputs , levels = encoder(input_shape)
    [f1 , f2 , f3 , f4, f5] = levels
    current_layer = f5
    # add levels with max pooling

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        current_layer = create_convolution_block(n_filters=levels[layer_depth]._keras_shape[1], kernel = (2,2) ,
                                                 input_layer=up_convolution, batch_normalization=batch_normalization)
        concat = concatenate([current_layer, levels[layer_depth]], axis=1)
        
        current_layer = create_convolution_block(n_filters=levels[layer_depth]._keras_shape[1], input_layer=concat,
                                                 batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        current_layer = SE_block(current_layer,levels[layer_depth]._keras_shape[1], ratio = 4, name='att'+str(layer_depth))

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

def unet_resnet_2d_4level(input_shape, encoder = get_resnet4level_encoder, final_act=None, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid"):

    
    n_labels = 4
    
    n_t = 1

    inputs , levels = encoder(input_shape)
    [f1 , f2 , f3 , f4] = levels
    current_layer = f4
    # add levels with max pooling

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        current_layer = create_convolution_block(n_filters=levels[layer_depth]._keras_shape[1], kernel = (2,2) ,
                                                 input_layer=up_convolution, batch_normalization=batch_normalization)
        concat = concatenate([current_layer, levels[layer_depth]], axis=1)
        
        current_layer = create_convolution_block(n_filters=levels[layer_depth]._keras_shape[1], input_layer=concat,
                                                 batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth]._keras_shape[1],
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
        


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def SE_block(x, out_dim, ratio, name, batch_norm=False):
    """
    self attention squeeze-excitation block, attention mechanism on channel dimension
    :param x: input feature map
    :return: attention weighted on channel dimension feature map
    """
    # Squeeze: global average pooling
    x_s = layers.GlobalAveragePooling2D(data_format=None)(x)
    # Excitation: bottom-up top-down FCs
    if batch_norm:
        x_s = layers.BatchNormalization()(x_s)
    x_e = layers.Dense(units=out_dim//ratio)(x_s)
    x_e = layers.Activation('relu')(x_e)
    if batch_norm:
        x_e = layers.BatchNormalization()(x_e)
    x_e = layers.Dense(units=out_dim)(x_e)
    x_e = layers.Activation('sigmoid')(x_e)
    x_e = layers.Reshape((out_dim, 1, 1), name=name+'channel_weight')(x_e)
    result = layers.multiply([x, x_e])
    return result



