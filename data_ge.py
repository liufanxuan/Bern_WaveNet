import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Add, Activation, Embedding, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import h5py
import tables
import pandas as pd
import time
import random
import shutil
import os
import numpy as np 
import keras.backend as K
from keras.callbacks import TensorBoard
K.set_image_data_format("channels_first")

from WMUnet_3d import unet_model_3d
def data_generator_new(data, name, index_all, dose_list, length_train, length_1st):
    if 1:
        while True:
            x_batch, y_batch = [], []
            count = 0
            for index in index_all:
                ni = 0
                j = (index-1)%length_train
                n = (index-1)//length_train
                if j >= length_1st:
                    ni = 1
                    j = j - length_1st

                print(name[ni]+dose_list[n]+str(j))
                x = data.get(name[ni]+dose_list[n])[j]
                y = data.get(name[ni]+'_normal')[j]
                if len(x.shape)<4:
                    x = np.expand_dims(x, axis=0)
                x_batch.append(x)
                if len(y.shape)<4:
                    y = np.expand_dims(y, axis=0)
                y_batch.append(y)
                count += 1
                if count == 4:
                    yield np.array(x_batch), np.array(y_batch)
                    count = 0
                    x_batch, y_batch = [], []
def data_generator(data, name, index_all, dose_list, length_train):
    if 1:
        while True:
            x_batch, y_batch = [], []
            count = 0
            for index in index_all:
                j = (index-1)%length_train
                n = (index-1)//length_train
                print(name+dose_list[n]+str(j))
                x = data.get(name+dose_list[n])[j]
                y = data.get(name+'_normal')[j]
                if len(x.shape)<4:
                    x = np.expand_dims(x, axis=0)
                x_batch.append(x)
                if len(y.shape)<4:
                    y = np.expand_dims(y, axis=0)
                y_batch.append(y)
                count += 1
                # if count == self.batch_size:
                if count == 4:
                    yield np.array(x_batch), np.array(y_batch)
                    count = 0
                    x_batch, y_batch = [], []
def data_generator_test(data, name, length):
    if 1:
        while True:
            x_batch, y_batch = [], []
            count = 0
            if 1:
                for j in range(length):
                    print(name+str(j))
                    x = data.get(name+'_reduce')[j]
                    y = data.get(name+'_normal')[j]
                    if len(x.shape)<4:
                        x = np.expand_dims(x, axis=0)
                    x_batch.append(x)
                    if len(y.shape)<4:
                        y = np.expand_dims(y, axis=0)
                    y_batch.append(y)
                    count += 1
                    if count == 4:
                        yield np.array(x_batch), np.array(y_batch)
                        count = 0
                        x_batch, y_batch = [], []
               
data_path = '/media/data/fanxuan/data/data_h5data/data_all_50.h5'
test_data_path = '/media/data/fanxuan/data/data_h5data/data_test.h5'
f = tables.open_file(data_path)
f.close()
data = h5py.File(data_path, mode='r')
test_data = h5py.File(test_data_path, mode='r')
train_filenames = np.array(data['train_filenames'])
valid_filenames = np.array(data['valid_filenames'])
test_filenames = np.array(test_data['test_filenames'])
length_train = len(train_filenames)
print(length_train)
length_all = len(train_filenames)+len(valid_filenames)
print(length_all)
#print(train_filenames[length_train])
index_all = np.arange(1,length_all*6+1)
random.shuffle(index_all)
dose_list = ['_2_dose', '_4_dose', '_10_dose', '_20_dose', '_50_dose', '_100_dose']

'''x = data.get('train'+dose_list[2])[1]
print(dose_list[(len(index_all)-1)//length_train])
print(np.shape(x))'''

gen = data_generator_new(data, ['train','valid'], index_all,dose_list,length_all, length_train)
gen2 = data_generator_test(test_data, 'test', len(test_filenames))
for i in range(5):
    print(i)
    for index in range(int(100/ 4)):
        x_batch, y_batch = next(gen2)
'''
data_path = '/media/data/fanxuan/data/data_h5data/data_wave_all.h5'
f = h5py.File(data_path, mode='r')
for group in f.keys():
    print(1)
    # group_read = f['train_2_dose']
'''  