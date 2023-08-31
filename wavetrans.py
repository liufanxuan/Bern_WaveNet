import math
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Add, UpSampling2D, Activation, BatchNormalization, PReLU
from keras.optimizers import Adam
from keras import layers
from keras.layers import ZeroPadding3D
from keras.layers.convolutional import Conv3D
from keras.models import Sequential, Model
import pywt
import pywt.data


#def shuffle_channel()

K.set_image_data_format("channels_first")
IMAGE_ORDERING = 'channels_first'


#def shuffle_channel()



def dwt(x, data_format='channels_last'):

    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """

    if data_format == 'channels_last':
        # [all samplesL, width, height, neurons]
        x1 = x[:, 0::2, 0::2, :]  # x(2i?1, 2j?1)
        x2 = x[:, 1::2, 0::2, :]  # x(2i, 2j-1)
        x3 = x[:, 0::2, 1::2, :]  # x(2i?1, 2j)
        x4 = x[:, 1::2, 1::2, :]  # x(2i, 2j)

    elif data_format == 'channels_first':
        x1 = x[:, :, 0::2, 0::2]  # x(2i?1, 2j?1)
        x2 = x[:, :, 1::2, 0::2]  # x(2i, 2j-1)
        x3 = x[:, :, 0::2, 1::2]  # x(2i?1, 2j)
        x4 = x[:, :, 1::2, 1::2]  # x(2i, 2j)

    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4

    if data_format == 'channels_last':
        return K.concatenate([x_LL, x_LH, x_HL, x_HH], axis=-1)
    elif data_format == 'channels_first':
        return K.concatenate([x_LL, x_LH, x_HL, x_HH], axis=1)
        
def dwt_3d(x, data_format='channels_first'):

    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """


    if data_format == 'channels_first':


        x10 = x[:, :, 0::2, 0::2,0::2]  # x(2i?1, 2j?1)
        x11 = x[:, :, 0::2, 0::2,1::2]  # x(2i?1, 2j?1)
        x20 = x[:, :, 1::2, 0::2,0::2]  # x(2i, 2j-1)
        x21 = x[:, :, 1::2, 0::2,1::2]  # x(2i, 2j-1)
        x30 = x[:, :, 0::2, 1::2,0::2]  # x(2i?1, 2j)
        x31 = x[:, :, 0::2, 1::2,1::2]  # x(2i?1, 2j)
        x40 = x[:, :, 1::2, 1::2,0::2]  # x(2i, 2j)
        x41 = x[:, :, 1::2, 1::2,1::2]  # x(2i, 2j)
    elif data_format == 'channels_last':


        x10 = x[:, 0::2, 0::2,0::2, :]  # x(2i?1, 2j?1)
        x11 = x[:, 0::2, 0::2,1::2, :]  # x(2i?1, 2j?1)
        x20 = x[:, 1::2, 0::2,0::2, :]  # x(2i, 2j-1)
        x21 = x[:, 1::2, 0::2,1::2, :]  # x(2i, 2j-1)
        x30 = x[:, 0::2, 1::2,0::2, :]  # x(2i?1, 2j)
        x31 = x[:, 0::2, 1::2,1::2, :]  # x(2i?1, 2j)
        x40 = x[:, 1::2, 1::2,0::2, :]  # x(2i, 2j)
        x41 = x[:, 1::2, 1::2,1::2, :]  # x(2i, 2j)
    
    x_LLL = x10 + x20 + x30 + x40 + x11 + x21 + x31 + x41
    x_LLH = x10 + x20 + x30 + x40 - x11 - x21 - x31 - x41
    x_LHL = x10 + x20 - x30 - x40 + x11 + x21 - x31 - x41
    x_LHH = x10 + x20 - x30 - x40 - x11 - x21 + x31 + x41
    x_HLL = x10 - x20 + x30 + x40 + x11 - x21 + x31 + x41
    x_HLH = x10 - x20 + x30 + x40 - x11 + x21 - x31 - x41
    x_HHL = x10 - x30 - x20 + x40 + x11 - x31 - x21 + x41
    x_HHH = x10 - x30 - x20 + x40 - x11 + x31 + x21 - x41

    if data_format == 'channels_last':
        return K.concatenate([x_LLL, x_LLH, x_LHL, x_LHH, x_HLL, x_HLH, x_HHL, x_HHH], axis=4)
    elif data_format == 'channels_first':
        return K.concatenate([x_LLL, x_LLH, x_LHL, x_LHH, x_HLL, x_HLH, x_HHL, x_HHH], axis=1)


def iwt(x, data_format='channels_last'):
    """
    IWT (Inverse Wavelet Transfomr) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """
    if data_format == 'channels_last':

        x_LL = x[:, :, :, 0:x.shape[3]//4]
        x_LH = x[:, :, :, x.shape[3]//4:x.shape[3]//4*2]
        x_HL = x[:, :, :, x.shape[3]//4*2:x.shape[3]//4*3]
        x_HH = x[:, :, :, x.shape[3]//4*3:]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4

        y1 = K.stack([x1, x3], axis=2)
        y2 = K.stack([x2, x4], axis=2)
        shape = K.shape(x)

        return K.reshape(K.concatenate([y1, y2], axis=-1), K.stack([shape[0], shape[1]*2, shape[2]*2, shape[3]//4]))

    elif data_format == 'channels_first':

        raise RuntimeError('WIP, please use "channels_last" instead.')

        x_LL = x[:, 0:x.shape[1]//4, :, :]
        x_LH = x[:, x.shape[1]//4:x.shape[1]//4*2, :, :]
        x_HL = x[:, x.shape[1]//4*2:x.shape[1]//4*3, :, :]
        x_HH = x[:, x.shape[1]//4*3:, :, :]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4

        y1 = K.stack([x1, x3], axis=3)
        y2 = K.stack([x2, x4], axis=3)
        shape = K.shape(x)
        return K.reshape(K.concatenate([y1, y2], axis=1), K.stack([shape[0], shape[1]//4, shape[2]*2, shape[3]*2]))


class DWT_Pooling(layers.Layer):
    """
    Custom Layer performing DWT pooling operation described in :
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    # Arguments :
        data_format (String): 'channels_first' or 'channels_last'
    # Input shape :
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows, cols, channels)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels, rows, cols)
    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows/2, cols/2, channels*4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels*4, rows/2, cols/2)
    """

    def __init__(self, data_format=None,**kwargs):
        super(DWT_Pooling, self).__init__(**kwargs)
        self.data_format = 'channels_last'

    def build(self, input_shape):
        super(DWT_Pooling, self).build(input_shape)

    def call(self, x):
        return dwt(x, self.data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1]*4, input_shape[2]//2, input_shape[3]//2)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, input_shape[3]*4)


class IWT_UpSampling(layers.Layer):
    """
    Custom Layer performing IWT upsampling operation described in :
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    # Arguments :
        data_format (String): 'channels_first' or 'channels_last'
    # Input shape :
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows, cols, channels)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels, rows, cols)
    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows*2, cols*2, channels/4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels/4, rows*2, cols*2)
    """

    def __init__(self, data_format=None, **kwargs):
        super(IWT_UpSampling, self).__init__(**kwargs)
        self.data_format = 'channels_last'

    def build(self, input_shape):
        super(IWT_UpSampling, self).build(input_shape)

    def call(self, x):
        return iwt(x, self.data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return ( input_shape[0], input_shape[1]//4, input_shape[2]*2, input_shape[3]*2 )
        elif self.data_format == 'channels_last':
            return ( input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3]//4 )

class DWT_3D_Pooling(layers.Layer):
    """
    Custom Layer performing DWT pooling operation described in :
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    # Arguments :
        data_format (String): 'channels_first' or 'channels_last'
    # Input shape :
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows, cols, channels)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels, rows, cols)
    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows/2, cols/2, channels*4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels*4, rows/2, cols/2)
    """

    def __init__(self, data_format=None,**kwargs):
        super(DWT_3D_Pooling, self).__init__(**kwargs)
        self.data_format = 'channels_first'

    def build(self, input_shape):
        super(DWT_3D_Pooling, self).build(input_shape)

    def call(self, x):
        return dwt_3d(x, self.data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1]*8, input_shape[2]//2, input_shape[3]//2, input_shape[4]//2)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, input_shape[3]//2, input_shape[4]*8)


class IWT_3D_UpSampling(layers.Layer):
    """
    Custom Layer performing IWT upsampling operation described in :
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    # Arguments :
        data_format (String): 'channels_first' or 'channels_last'
    # Input shape :
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows, cols, channels)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels, rows, cols)
    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows*2, cols*2, channels/4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels/4, rows*2, cols*2)
    """

    def __init__(self, data_format=None, **kwargs):
        super(IWT_UpSampling, self).__init__(**kwargs)
        self.data_format = 'channels_last'

    def build(self, input_shape):
        super(IWT_UpSampling, self).build(input_shape)

    def call(self, x):
        return iwt(x, self.data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return ( input_shape[0], input_shape[1]//4, input_shape[2]*2, input_shape[3]*2 )
        elif self.data_format == 'channels_last':
            return ( input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3]//4 )

class DWT_Pooling_Db4(layers.Layer):
    """
    Custom Layer performing DWT pooling operation with db4 :
    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows/2, cols/2, channels*4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels*4, rows/2, cols/2)
    """

    def __init__(self, **kwargs):
        super(DWT_Pooling_Db4, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DWT_Pooling_Db4, self).build(input_shape)

    def call(self, x):
        
        return db4_dwt(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]//2, input_shape[2]//2, input_shape[3]*4)


class IWT_UpSampling_Db4(layers.Layer):
    """
    Custom Layer performing IWT upsampling operation described in :
    """

    def __init__(self , **kwargs):
        super(IWT_UpSampling_Db4, self).__init__(**kwargs)

    def build(self, input_shape):
        super(IWT_UpSampling_Db4, self).build(input_shape)

    def call(self, x):
        return db4_iwt(x)

    def compute_output_shape(self, input_shape):
        return ( input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3]//4 )
        
        
def dwt_3d_sa(x, data_format='channels_first'):

    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """


    if data_format == 'channels_first':


        x10 = x[:, :, 0::2, 0::2,0::2]  # x(2i?1, 2j?1)
        x11 = x[:, :, 0::2, 0::2,1::2]  # x(2i?1, 2j?1)
        x20 = x[:, :, 1::2, 0::2,0::2]  # x(2i, 2j-1)
        x21 = x[:, :, 1::2, 0::2,1::2]  # x(2i, 2j-1)
        x30 = x[:, :, 0::2, 1::2,0::2]  # x(2i?1, 2j)
        x31 = x[:, :, 0::2, 1::2,1::2]  # x(2i?1, 2j)
        x40 = x[:, :, 1::2, 1::2,0::2]  # x(2i, 2j)
        x41 = x[:, :, 1::2, 1::2,1::2]  # x(2i, 2j)
    elif data_format == 'channels_last':


        x10 = x[:, 0::2, 0::2,0::2, :]  # x(2i?1, 2j?1)
        x11 = x[:, 0::2, 0::2,1::2, :]  # x(2i?1, 2j?1)
        x20 = x[:, 1::2, 0::2,0::2, :]  # x(2i, 2j-1)
        x21 = x[:, 1::2, 0::2,1::2, :]  # x(2i, 2j-1)
        x30 = x[:, 0::2, 1::2,0::2, :]  # x(2i?1, 2j)
        x31 = x[:, 0::2, 1::2,1::2, :]  # x(2i?1, 2j)
        x40 = x[:, 1::2, 1::2,0::2, :]  # x(2i, 2j)
        x41 = x[:, 1::2, 1::2,1::2, :]  # x(2i, 2j)
    
    x_LLL = x10 + x20 + x30 + x40 + x11 + x21 + x31 + x41
    x_LLH = x10 + x20 + x30 + x40 - x11 - x21 - x31 - x41
    x_LHL = x10 + x20 - x30 - x40 + x11 + x21 - x31 - x41
    x_LHH = x10 + x20 - x30 - x40 - x11 - x21 + x31 + x41
    x_HLL = x10 - x20 + x30 + x40 + x11 - x21 + x31 + x41
    x_HLH = x10 - x20 + x30 + x40 - x11 + x21 - x31 - x41
    x_HHL = x10 - x30 - x20 + x40 + x11 - x31 - x21 + x41
    x_HHH = x10 - x30 - x20 + x40 - x11 + x31 + x21 - x41
    x_LLL = spatial_attention(x_LLL)
    if data_format == 'channels_last':
        return K.concatenate([x_LLL, x_LLH, x_LHL, x_LHH, x_HLL, x_HLH, x_HHL, x_HHH], axis=4)
    elif data_format == 'channels_first':
        return K.concatenate([x_LLL, x_LLH, x_LHL, x_LHH, x_HLL, x_HLH, x_HHL, x_HHH], axis=1)

def iwt(x, data_format='channels_last'):
    """
    IWT (Inverse Wavelet Transfomr) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """
    if data_format == 'channels_last':

        x_LL = x[:, :, :, 0:x.shape[3]//4]
        x_LH = x[:, :, :, x.shape[3]//4:x.shape[3]//4*2]
        x_HL = x[:, :, :, x.shape[3]//4*2:x.shape[3]//4*3]
        x_HH = x[:, :, :, x.shape[3]//4*3:]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4

        y1 = K.stack([x1, x3], axis=2)
        y2 = K.stack([x2, x4], axis=2)
        shape = K.shape(x)

        return K.reshape(K.concatenate([y1, y2], axis=-1), K.stack([shape[0], shape[1]*2, shape[2]*2, shape[3]//4]))

    elif data_format == 'channels_first':

        raise RuntimeError('WIP, please use "channels_last" instead.')

        x_LL = x[:, 0:x.shape[1]//4, :, :]
        x_LH = x[:, x.shape[1]//4:x.shape[1]//4*2, :, :]
        x_HL = x[:, x.shape[1]//4*2:x.shape[1]//4*3, :, :]
        x_HH = x[:, x.shape[1]//4*3:, :, :]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4

        y1 = K.stack([x1, x3], axis=3)
        y2 = K.stack([x2, x4], axis=3)
        shape = K.shape(x)
        return K.reshape(K.concatenate([y1, y2], axis=1), K.stack([shape[0], shape[1]//4, shape[2]*2, shape[3]*2]))

class DWT_3D_Pooling_sa(layers.Layer):
    """
    Custom Layer performing DWT pooling operation described in :
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    # Arguments :
        data_format (String): 'channels_first' or 'channels_last'
    # Input shape :
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows, cols, channels)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels, rows, cols)
    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows/2, cols/2, channels*4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels*4, rows/2, cols/2)
    """

    def __init__(self, data_format=None,**kwargs):
        super(DWT_3D_Pooling_sa, self).__init__(**kwargs)
        self.data_format = 'channels_first'

    def build(self, input_shape):
        super(DWT_3D_Pooling_sa, self).build(input_shape)

    def call(self, x):
        x = dwt_3d_sa(x, self.data_format)
        return x
        
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1]*8, input_shape[2]//2, input_shape[3]//2, input_shape[4]//2)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, input_shape[3]//2, input_shape[4]*8)


def spatial_attention(inputs):
    print(inputs.shape[1])
    c = inputs.shape[1]
    x_1 = create_convolution_block(inputs, int(c//4), batch_normalization=False)  # 在通道维度求最大值
    x_2 = create_convolution_block(x_1, int(c//4), batch_normalization=False)
    x_3 = create_convolution_block(x_2, int(c//4), batch_normalization=False)

    # 在通道维度上堆叠[b,h,w,2]
    x = layers.concatenate([x_1, x_2, x_3], axis=1)
    # 1*1卷积调整通道[b,h,w,1]
    x = Conv3D(filters=int(c), kernel_size=(1,1,1), strides=1, padding='same')(x)
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