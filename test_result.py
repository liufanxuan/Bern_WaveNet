import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Add, Activation, Embedding, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import multi_gpu_model
from keras.layers.convolutional import Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import h5py
import atexit
import random
import pandas as pd
from tqdm import trange
import time
import os
import numpy as np 
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
K.set_image_data_format("channels_first")

from WMUnet_3d import unet_model_3d
 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tf.set_random_seed(1234)
random.seed(2)
class SR_UnetGAN():
    def __init__(self):
        self.data_path = '/media/data/fanxuan/data/data_h5data/data_all.h5'
        self.test_data_path = '/media/data/fanxuan/data/data_h5data/data_test.h5'
        #data_Siemens_wave_50_bior3_7_3d.h5
        self.save_dir = '/media/data/fanxuan/result/FCN_multi_4_3_3_new'
        #self.model_path = '/media/data/fanxuan/result/FCN_multi_4/model/generator_epoch_26.hdf5'
        self.img_shape = (8,176,176,16)
        #self.common_optimizer = AdamWOptimizer(weight_decay=1e-4, learning_rate=0.001)
        self.common_optimizer = Adam(0.0001, 0.5)
        self.epochs = 400
        self.batch_size = 4

        self.generator = self.build_generator()
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)
        self.generator2 = self.build_generator()
        self.generator2.compile(loss="mse", optimizer=self.common_optimizer)

    def build_generator(self, path=None):
        model = unet_model_3d(self.img_shape)
        if path is not None:
            model.load_weights(path)
            print('Success Loading Model!')
        #model.summary()
        return model

    def data_generator_new(self, data, name, index_all, dose_list, length_train, length_1st):
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

                # print(name+dose_list[n]+str(j))
                x = data.get(name[ni]+dose_list[n])[j]
                x = x*10
                y = data.get(name[ni]+'_normal')[j]
                y = y*10
                if len(x.shape)<4:
                    x = np.expand_dims(x, axis=0)
                x_batch.append(x)
                if len(y.shape)<4:
                    y = np.expand_dims(y, axis=0)
                y_batch.append(y)
                count += 1
                if count == self.batch_size:
                    yield np.array(x_batch), np.array(y_batch)
                    count = 0
                    x_batch, y_batch = [], []
    def data_generator_test(self, data, name, length):
        while True:
            x_batch, y_batch = [], []
            count = 0
            if 1:
                for j in range(length):
                    x = data.get(name+'_reduce')[j]
                    x = x*10
                    y = data.get(name+'_normal')[j]
                    y = y*10
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

    def write_log(self, callback, name, value, batch_no):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

    def train(self):
        data = h5py.File(self.data_path, mode='r')
        
        test_data = h5py.File(self.test_data_path, mode='r')
        
        train_filenames = np.array(data['train_filenames'])
        valid_filenames = np.array(data['valid_filenames'])
        length_train = len(train_filenames)+len(valid_filenames)
        index_all = np.arange(1,(len(train_filenames)+len(valid_filenames))*6+1)
        random.shuffle(index_all)
        dose_list = ['_2_dose', '_4_dose', '_10_dose', '_20_dose', '_50_dose', '_100_dose']
        
        data_generator = self.data_generator_new(data, ['train','valid'], index_all, dose_list, length_train, len(train_filenames))
        test_filenames = np.array(test_data['test_filenames'])
        valid_generator = self.data_generator_test(test_data, 'test', len(test_filenames))
        
        log_dir = os.path.join(self.save_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard.set_model(self.generator)
        batch_df = pd.DataFrame()
        epoch_df = pd.DataFrame()
        batch_val = pd.DataFrame()
        epoch_val = pd.DataFrame()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            sess.run(tf.global_variables_initializer())
            K.set_session(sess)
            def save_model():
                print('Saving model.')    
                self.generator.save_weights(os.path.join(self.save_dir, 'generator_latest.hdf5'))
            atexit.register(save_model)
            self.generator.load_weights('/media/data/fanxuan/result/FCN_multi_3_10/model/generator_epoch_1.hdf5')
            self.generator2.load_weights('/media/data/fanxuan/result/FCN_multi_3_10/model/generator_epoch_19.hdf5')
            if 1:
                val_losses = []
                val_losses2 = []
                ori_losses = []
                for ind in range(int(len(test_filenames) / self.batch_size)):
                # int(len(valid_filenames) / self.batch_size)
                    x_val, y_val = next(valid_generator)
                    ori_loss = np.mean(np.mean(np.square(np.array(y_val) - np.array(x_val)), axis=-1))
                    pred_val = self.generator.predict(x_val)
                    pred_val2 = self.generator2.predict(x_val)
                    a = np.mean(np.square(np.array(y_val) - np.array(pred_val)), axis=-1)
                    #print(a.shape)
                    val_loss = np.mean(a)
                    val_loss2 = np.mean(np.mean(np.square(np.array(y_val) - np.array(pred_val2)), axis=-1))
                    print("    val_loss: {}\n".format(val_loss))
                    print("    val_loss2: {}\n".format(val_loss2))
                    print("    ori_loss: {}\n".format(ori_loss))
                    ori_losses.append(ori_loss)
                    val_losses.append(val_loss)
                    val_losses2.append(val_loss2)
                
                
                print("val_loss: {}\n".format(np.mean(val_losses)))
                print("val_loss2: {}\n".format(np.mean(val_losses2)))
                print("ori_loss: {}\n".format(np.mean(ori_losses)))
'''
                batch_val = batch_val.append(pd.DataFrame({'epoch': [epoch] * len(val_losses),
                                                         'batch': np.arange(1, len(val_losses)+1),
                                                         'val_loss': val_losses}))
                epoch_val = epoch_val.append(pd.DataFrame({'epoch': [epoch],
                                                         'val_loss': [np.mean(val_losses)]}))
                batch_val = batch_val[['epoch', 'batch', 'val_loss']]
                epoch_val = epoch_val[['epoch', 'val_loss']]
                batch_val.to_csv(os.path.join(log_dir, 'batch_val_loss.csv'), index=False)
                epoch_val.to_csv(os.path.join(log_dir, 'epoch_val_loss.csv'), index=False)
                # Save losses to Tensorboard
                self.write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

                model_dir = os.path.join(self.save_dir, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                if self.epochs > 20:
                    if epoch % 10 == 0:
                        
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
                else:
                    if epoch % 5 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))'''

if __name__ == '__main__':
    
    gan = SR_UnetGAN()
    gan.train()