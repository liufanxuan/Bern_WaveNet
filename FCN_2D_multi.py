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

from Unet_multi import unet_model_2d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class SR_UnetGAN():
    def __init__(self):
        self.data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_10_wave.h5'
        self.save_dir = '/media/data/fanxuan/result/FCN_10_50_wave_test'
        self.img_shape = (4,176,176)
        self.common_optimizer = Adam(0.0002, 0.5)
        self.epochs = 400
        self.batch_size = 4

        self.generator = self.build_generator()
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)

    def build_generator(self):
        model = unet_model_2d(self.img_shape)
        model.summary()
        return model
    
    def data_generator(self, data, name, length):
        while True:
            x1_batch, x2_batch, y_batch = [], [], []
            count = 0
            for j in range(length):
                x1 = data.get(name+'_reduce_50')[j]
                x2 = data.get(name+'_reduce_10')[j]
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
                    x1_batch, x2_batch, y_batch = next(data_generator)
                    
                    [g_loss, g_loss1, g_loss2] = self.generator.train_on_batch(x1_batch,[x2_batch, y_batch])
                    gen_losses.append(g_loss)
                    gen_losses1.append(g_loss1)
                    gen_losses2.append(g_loss2)
                    print("    G_loss: {}\n".format(g_loss))
                    
                val_losses_x2, val_losses_final = [],[]
                for ind in range(int(len(valid_filenames) / self.batch_size)):
                # int(len(valid_filenames) / self.batch_size)
                    x1_val, x2_val, y_val = next(valid_generator)
                    [pred_x2, pred_y] = self.generator.predict(x1_val)
                    x2_loss = np.mean(np.square(np.array(pred_x2) - np.array(x2_val)), axis=-1)
                    final_loss = np.mean(np.square(np.array(y_val) - np.array(pred_y)), axis=-1)
                    final_loss = np.mean(final_loss)
                    x2_loss = np.mean(x2_loss)
                    print("    val_loss: {}\n".format(final_loss,x2_loss))
                    val_losses_final.append(final_loss)
                    val_losses_x2.append(x2_loss)

                batch_df = batch_df.append(pd.DataFrame({'epoch': [epoch] * len(gen_losses),
                                                         'batch': np.arange(1, len(gen_losses)+1),
                                                         'generator_loss': gen_losses,
                                                         'generator_loss_mid': gen_losses1,
                                                         'generator_loss_final': gen_losses2}))
                epoch_df = epoch_df.append(pd.DataFrame({'epoch': [epoch],
                                                         'generator_loss': [np.mean(gen_losses)],
                                                         'generator_loss_mid': [np.mean(gen_losses1)],
                                                         'generator_loss_final': [np.mean(gen_losses2)]}))
                batch_df = batch_df[['epoch', 'batch', 'generator_loss', 'generator_loss_mid', 'generator_loss_final']]
                epoch_df = epoch_df[['epoch', 'generator_loss', 'generator_loss_mid', 'generator_loss_final']]
                batch_df.to_csv(os.path.join(log_dir, 'batch_loss.csv'), index=False)
                epoch_df.to_csv(os.path.join(log_dir, 'epoch_loss.csv'), index=False)


                batch_val = batch_val.append(pd.DataFrame({'epoch': [epoch] * len(val_losses_final),
                                                         'batch': np.arange(1, len(val_losses_final)+1),
                                                         'val_loss_mid': val_losses_x2,
                                                         'val_loss_final': val_losses_final}))
                epoch_val = epoch_val.append(pd.DataFrame({'epoch': [epoch],
                                                         'val_loss_mid': [np.mean(val_losses_x2)],
                                                         'val_loss_final': [np.mean(val_losses_final)]}))
                batch_val = batch_val[['epoch', 'batch', 'val_loss_mid', 'val_loss_final']]
                epoch_val = epoch_val[['epoch',  'val_loss_mid', 'val_loss_final']]
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