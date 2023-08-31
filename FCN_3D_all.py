import tensorflow as tf
from tensorflow.contrib.opt import AdamWOptimizer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Add, Activation, Embedding, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import h5py
import random
import pandas as pd
import time
import atexit
import os
import numpy as np 
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
K.set_image_data_format("channels_first")

from WMUnet_3d import unet_model_3d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.set_random_seed(1234)
random.seed(2)
class SR_UnetGAN():
    def __init__(self):
        self.data_path = '/media/data/fanxuan/data/data_h5data/data_all.h5'
        #data_Siemens_wave_50_bior3_7_3d.h5
        self.save_dir = '/media/data/fanxuan/result/FCN_all_1e5'
        self.img_shape = (8,176,176,16)
        #self.common_optimizer = AdamWOptimizer(weight_decay=1e-4, learning_rate=0.00001)
        self.common_optimizer = Adam(0.00001, 0.5)
        self.epochs = 400
        self.batch_size = 4

        self.generator = self.build_generator()
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)

    def autosave(self):
        self.generator.save_weights(os.path.join(model_dir, 'generator_latest.hdf5'))

    def build_generator(self, path=None):
        model = unet_model_3d(self.img_shape)
        if path is not None:
            model.load_weights(path)
        model.summary()
        return model
    def data_generator(self, data, name, index_all, dose_list, length_train):
        while True:
            x_batch, y_batch = [], []
            count = 0
            for index in index_all:
                j = (index-1)%length_train
                n = (index-1)//length_train
                # print(name+dose_list[n]+str(j))
                x = data.get(name+dose_list[n])[j]
                y = data.get(name+'_normal')[j]
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
    def data_generator_old(self, data, name, length, dose_list):
        while True:
            x_batch, y_batch = [], []
            count = 0
            for dose in dose_list:
                for j in range(length):
                    x = data.get(name+dose)[j]
                    y = data.get(name+'_normal')[j]
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

    def write_log(self, callback, name, value, batch_no):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

    def train(self):
        data = h5py.File(self.data_path, mode='r')
        train_filenames = np.array(data['train_filenames'])
        length_train = len(train_filenames)
        index_all = np.arange(1,len(train_filenames)*6+1)
        random.shuffle(index_all)
        dose_list = ['_2_dose', '_4_dose', '_10_dose', '_20_dose', '_50_dose', '_100_dose']
        data_generator = self.data_generator(data, 'train', index_all, dose_list, length_train)
        valid_filenames = np.array(data['valid_filenames'])
        valid_generator = self.data_generator_old(data, 'valid', len(valid_filenames), dose_list)
        log_dir = os.path.join(self.save_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard.set_model(self.generator)
        batch_df = pd.DataFrame()
        epoch_df = pd.DataFrame()
        batch_val = pd.DataFrame()
        epoch_val = pd.DataFrame()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            sess.run(tf.global_variables_initializer())
            K.set_session(sess)
            # Add a loop, which will run for a specified number of epochs:
            for epoch in range(1, self.epochs+1):
                # Create two lists to store losses
                gen_losses = []
                number_of_batches = int(len(train_filenames) / self.batch_size)
                for index in range(number_of_batches):
                # number_of_batches
                    print('Epoch: {}/{}\n  Batch: {}/{}'.format(epoch, self.epochs, index+1, number_of_batches))
                    x_batch, y_batch = next(data_generator)
                    g_loss = self.generator.train_on_batch(x_batch, y_batch)
                    gen_losses.append(g_loss)
                    print("    G_loss: {}\n".format(g_loss))
                    #te = Test()
                    #te.generate_result_3d(self.generator,epoch)
                    
                val_losses = []
                for ind in range(int(len(valid_filenames) / self.batch_size)):
                # int(len(valid_filenames) / self.batch_size)
                    x_val, y_val = next(valid_generator)
                    pred_val = self.generator.predict(x_val)
                    a = np.mean(np.square(np.array(y_val) - np.array(pred_val)), axis=-1)
                    print(a.shape)
                    val_loss = np.mean(a)
                    print("    val_loss: {}\n".format(val_loss))
                    val_losses.append(val_loss)
                
                
                batch_df = batch_df.append(pd.DataFrame({'epoch': [epoch] * len(gen_losses),
                                                         'batch': np.arange(1, len(gen_losses)+1),
                                                         'generator_loss': gen_losses}))
                epoch_df = epoch_df.append(pd.DataFrame({'epoch': [epoch],
                                                         'generator_loss': [np.mean(gen_losses)]}))
                batch_df = batch_df[['epoch', 'batch', 'generator_loss']]
                epoch_df = epoch_df[['epoch', 'generator_loss']]
                batch_df.to_csv(os.path.join(log_dir, 'batch_loss.csv'), index=False)
                epoch_df.to_csv(os.path.join(log_dir, 'epoch_loss.csv'), index=False)


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
                if epoch > 20:
                    if epoch % 10 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
                        '''if epoch > 140:
                            eva = Evaluate()
                            eva.generate_result_3d(self.generator,epoch,self.save_dir)
                            te = Test()
                            te.generate_result_3d(self.generator,epoch,self.save_dir)'''
                else:
                    if epoch % 5 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
        data.close()

if __name__ == '__main__':
    gan = SR_UnetGAN()
    atexit.register(gan.autosave)
    gan.train()