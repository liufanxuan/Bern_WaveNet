import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import nibabel as nib
import os
import numpy as np
import pandas as pd
import keras.backend as K
import time
from skimage import measure
import logging
K.set_image_data_format("channels_first")
import tensorflow as tf

from FCN_2D import SR_UnetGAN
from WMUnet_3d import unet_model_3d
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import pywt
import pywt.data


save_dir = '/media/data/fanxuan/result/FCN_50_wave_mw_3d/'
model_path = os.path.join(save_dir, 'model/generator_epoch_300.hdf5') 
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
    sess.run(tf.global_variables_initializer())
    K.set_session(sess)
    generator = unet_model_3d((8,176,176,16))
    generator.load_weights(model_path)

