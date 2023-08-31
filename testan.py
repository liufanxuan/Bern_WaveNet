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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SR_UnetGAN():
    def __init__(self):
        self.data_path = '/media/data/fanxuan/data/PART2_h5data/data_all_4channel_wave.h5'
        self.save_dir = '/media/data/fanxuan/result/FCN_all_multi_cov'
        self.img_shape = (4,176,176)
        self.common_optimizer = Adam(0.0002, 0.5)
        self.epochs = 500
        self.batch_size = 4

        self.generator = self.build_generator()
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)

    def build_generator(self):
        model = unet_model_2d(self.img_shape)
        model.summary()
        return model
    
    def data_generator(self, data, name, length):
        while True:
            x1_batch, x2_batch, x3_batch, x4_batch, x5_batch, y_batch = [], [], [], [], [], []
            count = 0
            for j in range(length):
                x1 = data.get(name+'_reduce_50')[j]
                x2 = data.get(name+'_reduce_20')[j]
                x3 = data.get(name+'_reduce_10')[j]
                x4 = data.get(name+'_reduce_4')[j]
                x5 = data.get(name+'_reduce_2')[j]
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
                if len(x4.shape)<3:
                    x4 = np.expand_dims(x4, axis=0)
                x4_batch.append(x4)
                if len(x5.shape)<3:
                    x5 = np.expand_dims(x5, axis=0)
                x5_batch.append(x5)
                if len(y.shape)<3:
                    y = np.expand_dims(y, axis=0)
                y_batch.append(y)
                count += 1
                if count == self.batch_size:
                    yield np.array(x1_batch), np.array(x2_batch), np.array(x3_batch), np.array(x4_batch), np.array(x5_batch), np.array(y_batch)
                    count = 0
                    x1_batch, x2_batch, x3_batch, x4_batch, x5_batch, y_batch = [], [],[], [], [],[]

    def data_generator1(self, data, name, length):
        while True:
            x1_batch, x2_batch, y_batch = [], [], []
            count = 0
            for j in range(length):
                x1 = data.get(name+'_reduce_50')[j]
                x2 = data.get(name+'_reduce_20')[j]
                y = data.get(name+'_normal')[j]
                if len(x1.shape)<3:
                    x1 = np.expand_dims(x1, axis=0)
                x1_batch.append(x1)
                if len(x2.shape)<3:
                    x2 = np.expand_dims(x2, axis=0)
                x2_batch.append(x2)
                if len(y.shape)<3:
                    y = np.expand_dims(y, axis=0)
                y_batch.append(y)
                count += 1
                if count == self.batch_size:
                    yield np.array(x1_batch), np.array(x2_batch), np.array(y_batch)
                    count = 0
                    x1_batch, x2_batch, y_batch = [], [],[]
                    
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
        data_generator = self.data_generator(data, 'train', len(train_filenames))
        valid_filenames = np.array(data['valid_filenames'])
        valid_generator = self.data_generator(data, 'valid', len(valid_filenames))
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
                gen_losses, gen_losses1, gen_losses2 = [],[],[]
                number_of_batches = int(len(train_filenames) / self.batch_size)
                for index in range(number_of_batches):
                # number_of_batches
                    print('Epoch: {}/{}\n  Batch: {}/{}'.format(epoch, self.epochs, index+1, number_of_batches))
                    x1_batch, x2_batch,x3_batch, x4_batch, x5_batch, y_batch = next(data_generator)
                    if epoch%5 == 1:
                        g_loss = self.generator.train_on_batch(x1_batch,y_batch)
                    elif epoch%5 == 2:
                        g_loss = self.generator.train_on_batch(x2_batch,y_batch)
                    elif epoch%5 == 3:
                        g_loss = self.generator.train_on_batch(x3_batch,y_batch)
                    elif epoch%5 == 4:
                        g_loss = self.generator.train_on_batch(x4_batch,y_batch)
                    elif epoch%5 == 0:
                        g_loss = self.generator.train_on_batch(x5_batch,y_batch)
                    
                    gen_losses.append(g_loss)
                    print("    G_loss: {}\n".format(g_loss))
                    
                val_losses_x1,  val_losses_x2,  val_losses_x3,  val_losses_x4,  val_losses_x5 = [],[],[],[],[]
                for ind in range(int(len(valid_filenames) / self.batch_size)):
                # int(len(valid_filenames) / self.batch_size)
                    x1_val, x2_val, x3_val, x4_val, x5_val, y_val = next(valid_generator)
                    pred_y = self.generator.predict(x1_val)
                    x1_loss = np.mean(np.square(np.array(pred_y) - np.array(y_val)), axis=-1)
                    x1_loss = np.mean(x1_loss)
                    print("    val_loss: {}\n".format(x1_loss))
                    val_losses_x1.append(x1_loss)
                    
                    pred_y = self.generator.predict(x2_val)
                    x2_loss = np.mean(np.square(np.array(pred_y) - np.array(y_val)), axis=-1)
                    x2_loss = np.mean(x2_loss)
                    print("    val_loss: {}\n".format(x2_loss))
                    val_losses_x2.append(x2_loss)
                    
                    pred_y = self.generator.predict(x3_val)
                    x3_loss = np.mean(np.square(np.array(pred_y) - np.array(y_val)), axis=-1)
                    x3_loss = np.mean(x3_loss)
                    print("    val_loss: {}\n".format(x3_loss))
                    val_losses_x3.append(x3_loss)
                    
                    pred_y = self.generator.predict(x4_val)
                    x4_loss = np.mean(np.square(np.array(pred_y) - np.array(y_val)), axis=-1)
                    x4_loss = np.mean(x4_loss)
                    print("    val_loss: {}\n".format(x4_loss))
                    val_losses_x4.append(x4_loss)
                    
                    pred_y = self.generator.predict(x5_val)
                    x5_loss = np.mean(np.square(np.array(pred_y) - np.array(y_val)), axis=-1)
                    x5_loss = np.mean(x5_loss)
                    print("    val_loss: {}\n".format(x5_loss))
                    val_losses_x5.append(x5_loss)

                batch_df = batch_df.append(pd.DataFrame({'epoch': [epoch] * len(gen_losses),
                                                         'batch': np.arange(1, len(gen_losses)+1),
                                                         'generator_loss': gen_losses}))
                epoch_df = epoch_df.append(pd.DataFrame({'epoch': [epoch],
                                                         'generator_loss': [np.mean(gen_losses)]}))
                batch_df = batch_df[['epoch', 'batch', 'generator_loss']]
                epoch_df = epoch_df[['epoch', 'generator_loss']]
                batch_df.to_csv(os.path.join(log_dir, 'batch_loss.csv'), index=False)
                epoch_df.to_csv(os.path.join(log_dir, 'epoch_loss.csv'), index=False)


                batch_val = batch_val.append(pd.DataFrame({'epoch': [epoch] * len(val_losses_x1),
                                                         'batch': np.arange(1, len(val_losses_x1)+1),
                                                         'val_loss_x1': val_losses_x1,
                                                         'val_loss_x2': val_losses_x2,
                                                         'val_loss_x3': val_losses_x3,
                                                         'val_loss_x4': val_losses_x4,
                                                         'val_loss_x5': val_losses_x5}))
                epoch_val = epoch_val.append(pd.DataFrame({'epoch': [epoch],
                                                         'val_loss_x1': [np.mean(val_losses_x1)],
                                                         'val_loss_x2': [np.mean(val_losses_x2)],
                                                         'val_loss_x3': [np.mean(val_losses_x3)],
                                                         'val_loss_x4': [np.mean(val_losses_x4)],
                                                         'val_loss_x5': [np.mean(val_losses_x5)]}))
                batch_val = batch_val[['epoch', 'batch', 'val_loss_x1', 'val_loss_x2','val_loss_x3','val_loss_x4','val_loss_x5']]
                epoch_val = epoch_val[['epoch',  'val_loss_x1', 'val_loss_x2','val_loss_x3','val_loss_x4','val_loss_x5']]
                batch_val.to_csv(os.path.join(log_dir, 'batch_val_loss.csv'), index=False)
                epoch_val.to_csv(os.path.join(log_dir, 'epoch_val_loss.csv'), index=False)
                # Save losses to Tensorboard
                self.write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

                model_dir = os.path.join(self.save_dir, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if self.epochs > 20:
                    if epoch % 10 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
                else:
                    if epoch % 5 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
        data.close()

if __name__ == '__main__':
    gan = SR_UnetGAN()
    gan.train()