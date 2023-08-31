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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf.set_random_seed(1234)
random.seed(2)
class SR_UnetGAN():
    def __init__(self):
        self.data_path = '/media/data/fanxuan/data/data_h5data/data_wave_uExplorer.h5'
        self.test_data_path = '/media/data/fanxuan/data/data_h5data/data_wave_test_uexplorer.h5'
        self.test_data_path2 = '/media/data/fanxuan/data/data_h5data/data_wave_test_real.h5'
        self.test_data_path3 = '/media/data/fanxuan/data/data_h5data/data_wave_test_subject.h5'
        #data_Siemens_wave_50_bior3_7_3d.h5
        self.save_dir = '/media/data/fanxuan/result/FCN_multiwave_uExplorer2_1E-4'
        #self.model_path = '/media/data/fanxuan/result/FCN_multi_4/model/generator_epoch_26.hdf5'
        self.img_shape = (8,176,176,16)
        #self.common_optimizer = AdamWOptimizer(weight_decay=1e-4, learning_rate=0.001)
        self.common_optimizer = Adam(0.00004, 0.5)
        self.epochs = 400
        self.batch_size = 4

        self.generator = self.build_generator()
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)
        #self.generator_parallel = multi_gpu_model(self.generator,2)
        #self.generator_parallel.compile(loss="mse", optimizer=self.common_optimizer)
        #self.generator_parallel.summary()

    def build_generator(self, path=None):
        model = unet_model_3d(self.img_shape)
        if path is not None:
            model.load_weights(path)
            print('Success Loading Model!')
        model.summary()
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
    def data_generator_test(self, data, name, index):
        while True:
            x_batch, y_batch = [], []
            count = 0
            if 1:
                for j in index:
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
    def data_generator_100(self, data, name, index_all, dose_list, length_train, length_1st):
        while True:
            x_batch, y_batch = [], []
            count = 0
            for index in index_all:
                ni = 0
                j = index
                if index >= length_1st:
                    ni = 1
                    j = j - length_1st

                # print(name+dose_list[n]+str(j))
                x = data.get(name[ni]+dose_list)[j]
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
        test_data2 = h5py.File(self.test_data_path2, mode='r')
        test_data3 = h5py.File(self.test_data_path3, mode='r')
        
        train_filenames = np.array(data['train_filenames'])
        valid_filenames = np.array(data['valid_filenames'])
        length_train = len(train_filenames)+len(valid_filenames)
        index_all = np.arange(1,(len(train_filenames)+len(valid_filenames))*6+1)
        random.shuffle(index_all)
        dose_list = ['_2_dose', '_4_dose', '_10_dose', '_20_dose', '_50_dose', '_100_dose']
        
        data_generator = self.data_generator_new(data, ['train','valid'], index_all, dose_list, length_train, len(train_filenames))
        #data_generator = self.data_generator_100(data, ['train','valid'], index_all, '_reduce', length_train, len(train_filenames))
        test_filenames = np.array(test_data['test_filenames'])
        index_all2 = np.arange(0,len(test_filenames))
        random.shuffle(index_all2)
        valid_generator = self.data_generator_test(test_data, 'test', index_all2)


        test_filenames2 = np.array(test_data2['test_filenames'])
        index_all22 = np.arange(0,len(test_filenames2))
        random.shuffle(index_all22)
        valid_generator2 = self.data_generator_test(test_data2, 'test', index_all22)


        test_filenames3 = np.array(test_data3['test_filenames'])
        index_all23 = np.arange(0,len(test_filenames3))
        random.shuffle(index_all23)
        valid_generator3 = self.data_generator_test(test_data3, 'test', index_all23)
        
        log_dir = os.path.join(self.save_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard.set_model(self.generator)
        batch_df = pd.DataFrame()
        epoch_df = pd.DataFrame()
        batch_val = pd.DataFrame()
        epoch_val = pd.DataFrame()
        epoch_ori = pd.DataFrame()
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
            self.generator.load_weights("/media/data/fanxuan/result/FCN_multiwave_uExplorer2/model/generator_epoch_15.hdf5")
            # Add a loop, which will run for a specified number of epochs:
            for epoch in range(1, self.epochs+1):
                gen_losses = []
                number_of_batches = int(len(index_all) / self.batch_size)
                for index in trange(number_of_batches):
                # number_of_batches
                    print('Epoch: {}/{}\n  Batch: {}/{}'.format(epoch, self.epochs, index+1, number_of_batches))
                    x_batch, y_batch = next(data_generator)
                    #ori_loss = np.mean(np.square(np.array(y_batch) - np.array(x_batch)))
                    #print("    ori_loss: {}\n".format(ori_loss))
                    #g_loss = self.generator_parallel.train_on_batch(x_batch, y_batch)
                    g_loss = self.generator.train_on_batch(x_batch, y_batch)
                    gen_losses.append(g_loss)
                    print("    G_loss: {}\n".format(g_loss))
                # Create two lists to store losses
                val_losses = []
                ori_losses = []
                val_losses2 = []
                ori_losses2 = []
                val_losses3 = []
                ori_losses3 = []

                for ind in range(int(len(test_filenames) / 4)):
                # int(len(valid_filenames) / 4)
                    x_val, y_val = next(valid_generator)
                    pred_val = self.generator.predict(x_val)
                    val_loss = np.mean(np.square(np.array(y_val) - np.array(pred_val)))
                    ori_loss = np.mean(np.square(np.array(y_val) - np.array(x_val)))
                    print("    val_loss: {}\n".format(val_loss))
                    print("    ori_loss: {}\n".format(ori_loss))
                    val_losses.append(val_loss)
                    ori_losses.append(ori_loss)
                for ind in range(int(len(test_filenames2) / 4)):
                # int(len(valid_filenames) / 4)
                    x_val, y_val = next(valid_generator2)
                    pred_val = self.generator.predict(x_val)
                    val_loss = np.mean(np.square(np.array(y_val) - np.array(pred_val)))
                    ori_loss = np.mean(np.square(np.array(y_val) - np.array(x_val)))
                    print("    val_loss: {}\n".format(val_loss))
                    print("    ori_loss: {}\n".format(ori_loss))
                    val_losses2.append(val_loss)
                    ori_losses2.append(ori_loss)
                for ind in range(int(len(test_filenames3) / 4)):
                # int(len(valid_filenames) / 4)
                    x_val, y_val = next(valid_generator3)
                    pred_val = self.generator.predict(x_val)
                    val_loss = np.mean(np.square(np.array(y_val) - np.array(pred_val)))
                    ori_loss = np.mean(np.square(np.array(y_val) - np.array(x_val)))
                    print("    val_loss: {}\n".format(val_loss))
                    print("    ori_loss: {}\n".format(ori_loss))
                    val_losses3.append(val_loss)
                    ori_losses3.append(ori_loss)
                    

                    #te = Test()
                    #te.generate_result_3d(self.generator,epoch)
                    

                
                
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
                                                         'val_loss_uexplorer': [np.mean(val_losses)],
                                                         'val_loss_real': [np.mean(val_losses2)],
                                                         'val_loss_subject': [np.mean(val_losses3)]}))
                batch_val = batch_val[['epoch', 'batch', 'val_loss']]
                epoch_val = epoch_val[['epoch', 'val_loss_uexplorer', 'val_loss_real', 'val_loss_subject']]
                batch_val.to_csv(os.path.join(log_dir, 'batch_val_loss.csv'), index=False)
                epoch_val.to_csv(os.path.join(log_dir, 'epoch_val_loss.csv'), index=False)


                epoch_ori = epoch_ori.append(pd.DataFrame({'epoch': [epoch],
                                                         'ori_loss_uexplorer': [np.mean(ori_losses)],
                                                         'ori_loss_real': [np.mean(ori_losses2)],
                                                         'ori_loss_subject': [np.mean(ori_losses3)]}))
                epoch_ori = epoch_ori[['epoch', 'ori_loss_uexplorer', 'ori_loss_real', 'ori_loss_subject']]
                epoch_ori.to_csv(os.path.join(log_dir, 'epoch_ori_loss.csv'), index=False)
                # Save losses to Tensorboard
                self.write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

                model_dir = os.path.join(self.save_dir, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                '''if self.epochs > 20:
                    if epoch % 10 == 0:
                        
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
                else:
                    if epoch % 5 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))'''
        data.close()

if __name__ == '__main__':
    
    gan = SR_UnetGAN()
    gan.train()