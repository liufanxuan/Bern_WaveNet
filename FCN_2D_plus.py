import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import h5py
import pandas as pd
import time
import os
import numpy as np 
import keras.backend as K
from keras.callbacks import TensorBoard
K.set_image_data_format("channels_first")

from Unet import unet_model_2d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SR_UnetGAN():
    def __init__(self):
        self.data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_10_4_wave.h5'
        self.save_dir = '/media/data/fanxuan/result/FCN_10_50_4_wave_new'
        self.img_shape1 = (4,176,176)
        self.img_shape2 = (8,176,176)
        self.img_shape3 = (12,176,176)
        self.common_optimizer = Adam(0.0002, 0.5)
        self.epochs = 200
        self.batch_size = 4

        # self.generator = self.build_generator()
        self.generator1 = self.build_generator1()
        self.generator2 = self.build_generator2()
        self.generator3 = self.build_generator3()
        self.generator1.compile(loss="mse", optimizer=self.common_optimizer)
        self.generator2.compile(loss="mse", optimizer=self.common_optimizer) 
        self.generator3.compile(loss="mse", optimizer=self.common_optimizer)        

    def build_generator1(self):
        model = unet_model_2d(self.img_shape1)
        return model
    def build_generator2(self):
        model = unet_model_2d(self.img_shape2)
        return model
    def build_generator3(self):
        model = unet_model_2d(self.img_shape3)
        return model  
          
    def data_generator2(self, data, name, length):
        while True:
            x1_batch, y_batch, x2_batch, x3_batch = [], [], [], []
            count = 0
            for j in range(length):
                x1 = data.get(name+'_reduce_50')[j]
                x2 = data.get(name+'_reduce_10')[j]
                x3 = data.get(name+'_reduce_4')[j]
                y = data.get(name+'_normal')[j]
                if len(x1.shape)<3:
                    x1 = np.expand_dims(x1, axis=0)
                x1_batch.append(x1)
                if len(x2.shape)<3:
                    x2 = np.expand_dims(x2, axis=0)
                x2_batch.append(x2)
                if len(x3.shape)<3:
                    x3 = np.expand_dims(x3, axis=0)
                x3_batch.append(x3)
                if len(y.shape)<3:
                    y = np.expand_dims(y, axis=0)
                y_batch.append(y)
                count += 1
                if count == self.batch_size:
                    yield np.array(x1_batch), np.array(x2_batch), np.array(x3_batch), np.array(y_batch)
                    count = 0
                    x1_batch, x2_batch, x3_batch, y_batch = [], [], [], []

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
        data_generator = self.data_generator2(data, 'train', len(train_filenames))
        valid_filenames = np.array(data['valid_filenames'])
        valid_generator = self.data_generator2(data, 'valid', len(valid_filenames))
        log_dir = os.path.join(self.save_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard1 = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard1.set_model(self.generator1)
        tensorboard2 = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard2.set_model(self.generator2)
        tensorboard3 = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard3.set_model(self.generator3)
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
                gen_losses1 = []
                gen_losses2 = []
                gen_losses3 = []
                number_of_batches = int(len(train_filenames) / self.batch_size)
                for index in range(number_of_batches):
                # number_of_batches
                    print('Epoch: {}/{}\n  Batch: {}/{}'.format(epoch, self.epochs, index+1, number_of_batches))
                    x1_batch, x2_batch, x3_batch, y_batch = next(data_generator)
                    g_loss1 = self.generator1.train_on_batch(x1_batch, x2_batch)
                    gen_losses1.append(g_loss1)
                    x_pred = self.generator1.predict(x1_batch)
                    x_con = np.concatenate((x_pred,x1_batch), axis=1)
                    print(x_con.shape)
                    g_loss2 = self.generator2.train_on_batch(x_con, x3_batch)
                    gen_losses2.append(g_loss2)
                    x_pred2 = self.generator2.predict(x_con)
                    x_3con = np.concatenate((x_pred2, x_con), axis=1)
                    g_loss3 = self.generator3.train_on_batch(x_3con, y_batch)
                    gen_losses3.append(g_loss3)
                    print("    G_loss:{}, {}, {}\n".format(g_loss1,g_loss2,g_loss3))
                    
                val_losses1,val_losses2,val_losses3 = [],[],[]
                for ind in range(int(len(valid_filenames) / self.batch_size)):
                # int(len(valid_filenames) / self.batch_size)
                    x1_val,x2_val,x3_val, y_val = next(valid_generator)
                    pred1_val = self.generator1.predict(x1_val)
                    a = np.mean(np.square(np.array(x2_val) - np.array(pred1_val)), axis=-1)
                    x_con = np.concatenate((pred1_val,x1_val), axis=1)
                    pred2_val = self.generator2.predict(x_con)
                    a2 = np.mean(np.square(np.array(x3_val) - np.array(pred2_val)), axis=-1)
                    x_con3 = np.concatenate((pred2_val,x_con), axis=1)
                    pred3_val = self.generator3.predict(x_con3)
                    a3 = np.mean(np.square(np.array(y_val) - np.array(pred3_val)), axis=-1)
                    print(a.shape)
                    val_loss1 = np.mean(a)
                    val_loss2 = np.mean(a2)
                    val_loss3 = np.mean(a3)
                    print("    val_loss: {},{},{}\n".format(val_loss1,val_loss2,val_loss3))
                    val_losses1.append(val_loss1)
                    val_losses2.append(val_loss2)
                    val_losses3.append(val_loss3)
                batch_df = batch_df.append(pd.DataFrame({'epoch': [epoch] * len(gen_losses1),
                                                         'batch': np.arange(1, len(gen_losses1)+1),
                                                         'generator_loss1': gen_losses1,
                                                         'generator_loss2': gen_losses2,
                                                         'generator_final': gen_losses3}))
                epoch_df = epoch_df.append(pd.DataFrame({'epoch': [epoch],
                                                         'generator_loss1': [np.mean(gen_losses1)],
                                                         'generator_loss2': [np.mean(gen_losses2)],
                                                         'generator_final': [np.mean(gen_losses3)]}))
                batch_df = batch_df[['epoch', 'batch', 'generator_loss1','generator_loss2','generator_final']]
                epoch_df = epoch_df[['epoch', 'generator_loss1','generator_loss2','generator_final']]
                batch_df.to_csv(os.path.join(log_dir, 'batch_loss.csv'), index=False)
                epoch_df.to_csv(os.path.join(log_dir, 'epoch_loss.csv'), index=False)


                batch_val = batch_val.append(pd.DataFrame({'epoch': [epoch] * len(val_losses1),
                                                         'batch': np.arange(1, len(val_losses1)+1),
                                                         'val_loss1': val_losses1,
                                                         'val_loss2': val_losses2,
                                                         'val_final': val_losses3}))
                epoch_val = epoch_val.append(pd.DataFrame({'epoch': [epoch],
                                                         'val_loss1': [np.mean(val_losses1)],
                                                         'val_loss2': [np.mean(val_losses2)],
                                                         'val_final': [np.mean(val_losses3)]}))
                batch_val = batch_val[['epoch', 'batch', 'val_loss1','val_loss2','val_final']]
                epoch_val = epoch_val[['epoch', 'val_loss1','val_loss2','val_final']]
                batch_val.to_csv(os.path.join(log_dir, 'batch_val_loss.csv'), index=False)
                epoch_val.to_csv(os.path.join(log_dir, 'epoch_val_loss.csv'), index=False)
                # Save losses to Tensorboard
                self.write_log(tensorboard1, 'generator_loss1', np.mean(gen_losses1), epoch)
                self.write_log(tensorboard2, 'generator_loss2', np.mean(gen_losses2), epoch)
                self.write_log(tensorboard3, 'generator_loss3', np.mean(gen_losses3), epoch)

                model_dir = os.path.join(self.save_dir, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if self.epochs > 20:
                    if epoch % 10 == 0:
                        self.generator1.save_weights(os.path.join(model_dir, 'generator1_epoch_{}.hdf5'.format(epoch)))
                        self.generator2.save_weights(os.path.join(model_dir, 'generator2_epoch_{}.hdf5'.format(epoch)))
                        self.generator3.save_weights(os.path.join(model_dir, 'generator3_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
                else:
                    if epoch % 5 == 0:
                        self.generator1.save_weights(os.path.join(model_dir, 'generator1_epoch_{}.hdf5'.format(epoch)))
                        self.generator2.save_weights(os.path.join(model_dir, 'generator2_epoch_{}.hdf5'.format(epoch)))
                        self.generator3.save_weights(os.path.join(model_dir, 'generator3_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
        data.close()

if __name__ == '__main__':
    gan = SR_UnetGAN()
    gan.train()