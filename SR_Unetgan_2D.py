import os
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import h5py
import time
import pandas as pd
import keras.backend as K
from keras.callbacks import TensorBoard
tf.disable_v2_behavior() 
K.set_image_data_format("channels_first")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Unet import unet_model_2d

class SR_UnetGAN():
    def __init__(self):
        drf =20
        fold = 5
        # self.data_path = '/media/data/uni/Dose_reduction/data/low_dose_converted/'\
        #                  'Train_generalize_brain_crossvalidation_shuffled/Train_generalize_brain_crossvalidation/fold_{}_shuffled.h5'.format(fold)
        # self.save_dir = '/media/data/uni/Dose_reduction/result/'\
        #                 'Train_generalize_brain_crossvalidation/fold_{}'.format(fold)
        self.data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_10_wave.h5'
        self.save_dir = '/media/data/fanxuan/result/FCN_50_wave_gan'
        # self.data_path = '/media/data/uni/Dose_reduction/data/low_dose_converted/Train_all_generalize_brain_include1_shuffled/Train_all_generalize_brain_include1/data_shuffled.h5'
        # self.save_dir = '/media/data/uni/Dose_reduction/result/Train_all_generalize_brain_include1_bs4_epoch100_7_6'
        self.img_shape = (4,176,176)
        self.common_optimizer = Adam(0.0002, 0.5)
        self.epochs = 300
        self.batch_size = 4

        '''
        self.generator = self.build_generator()
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=self.common_optimizer, metrics=['accuracy'])
        
        self.adversarial_model = self.build_adversarial_model()
        self.adversarial_model.compile(loss=['binary_crossentropy', 'mse'], 
                                        loss_weights=[1, 1000], 
                                        optimizer=self.common_optimizer)

        
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        '''
        #######
        optimizer = Adam(0.0002, 0.5)

        base_generator = self.build_generator()
        base_discriminator = self.build_discriminator()
        ########
        self.generator = Model(
            inputs=base_generator.inputs,
            outputs=base_generator.outputs)
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)
        self.discriminator = Model(
            inputs=base_discriminator.inputs,
            outputs=base_discriminator.outputs)
        
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer, metrics=['accuracy'])
 
        frozen_D = Model(
            inputs=base_discriminator.inputs,
            outputs=base_discriminator.outputs)
        frozen_D.trainable = False
        z = Input(shape=self.img_shape)
        img = self.generator(z)
        valid = frozen_D(img)
        self.adversarial_model =Model([z],[valid,img])
        self.adversarial_model.compile(loss=['binary_crossentropy', 'mse'], 
                                        loss_weights=[1, 1000], 
                                        optimizer=self.common_optimizer)
                                     

        # Then, create and compile the adversarial model
        

    def build_generator(self):

        model = unet_model_2d(self.img_shape)
        # model = unet_model_2d(self.img_shape, final_act='tanh')

        return model

    def build_discriminator(self):
        leakyrelu_alpha = 0.2
        momentum = 0.8
 
        input_shape = self.img_shape
        input_layer = Input(shape=input_shape)
 
        dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
        dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

        # Add the 2nd convolution block
        dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
        dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
        dis2 = BatchNormalization(momentum=momentum)(dis2)

        # Add the third convolution block
        dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
        dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
        dis3 = BatchNormalization(momentum=momentum)(dis3)
        dis8 = Flatten()(dis3)
        dis9 = Dense(units=256)(dis8)
        dis9 = LeakyReLU(alpha=0.2)(dis9)
        
        output = Dense(units=1, activation='sigmoid')(dis9)
        

        model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
        print(model.summary())
        return model

        


    def build_adversarial_model(self):

        input_A = Input(shape=self.img_shape)
        # input_B = Input(shape=self.img_shape)

        generated_B = self.generator(input_A)

        # Make the discriminator network as non-trainable
        self.discriminator.trainable = False

        # Get the probability of generated high-resolution images
        probs = self.discriminator(generated_B)

        # Create and compile an adversarial model
        model = Model([input_A], [probs, generated_B])
        model.compile(loss=['binary_crossentropy', 'mse'], 
                                        loss_weights=[1, 1000], 
                                        optimizer=self.common_optimizer)
        return model

    def data_generator(self, data, name, length):
        while True:
            x_batch, y_batch = [], []
            count = 0
            for j in range(length):
                x = data.get(name+'_reduce_50')[j]
                y = data.get(name+'_normal')[j]
                if len(x.shape)<3:
                    x = np.expand_dims(x, axis=0)
                x_batch.append(x)
                if len(y.shape)<3:
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
        data_generator = self.data_generator(data, 'train', len(train_filenames))
        valid_filenames = np.array(data['valid_filenames'])
        valid_generator = self.data_generator(data, 'valid', len(valid_filenames))

        log_dir = os.path.join(self.save_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard.set_model(self.generator)
        tensorboard.set_model(self.discriminator)
        tensorboard.set_model(self.adversarial_model)
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
                gen_losses, dis_losses, adv_losses = [],[],[]
                number_of_batches = int(len(train_filenames) / self.batch_size)
                for index in range(number_of_batches):
                    print('Epoch: {}/{}\n  Batch: {}/{}'.format(epoch, self.epochs, index+1, number_of_batches))
                    x_batch, y_batch = next(data_generator)
                    gen_x = self.generator.predict(x_batch,verbose=3)
                    # Make the discriminator network trainable
                    # self.discriminator.trainable = True
                    
                    
                               
                    # Create fake and real labels
                    labels_real = np.ones((self.batch_size ,1))
                    labels_fake = np.zeros((self.batch_size ,1))
                            
                    # Train the discriminator network
                    loss_real = self.discriminator.train_on_batch(y_batch, labels_real)
                    
                    print(self.discriminator.trainable)
                    print( "D_loss_real: {}\n ".format(loss_real))
                    loss_fake = self.discriminator.train_on_batch(gen_x, labels_fake)
                    
                    print(self.discriminator.trainable)
                    print( "D_loss_fake: {}\n ".format(loss_fake)) 

                           
                    # Calculate total discriminator loss
                    d_loss = 0.5 *  np.add(loss_real, loss_fake)
                    print( "D_loss: {}\n ".format(d_loss)) 

                    # Train the adversarial model
                    
                    a_loss, prob_loss, g_loss = self.adversarial_model.train_on_batch([x_batch],
                                                                   [labels_real, y_batch])
                    print(self.discriminator.trainable)
                    gen_losses.append(g_loss)
                    dis_losses.append(d_loss)
                    adv_losses.append(a_loss)
                    print("    Pro_loss: {}\n    G_loss: {}\n    D_loss: {}\n    Adv_loss: {}".format(prob_loss,g_loss, d_loss, a_loss))
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
                                                         'generator_loss': gen_losses, 
                                                         'discriminator_loss': dis_losses, 
                                                         'adversarial_loss': adv_losses}))
                epoch_df = epoch_df.append(pd.DataFrame({'epoch': [epoch],
                                                         'generator_loss': [np.mean(gen_losses)], 
                                                         'discriminator_loss': [np.mean(dis_losses)], 
                                                         'adversarial_loss': [np.mean(adv_losses)]}))
                batch_df = batch_df[['epoch', 'batch', 'generator_loss', 'discriminator_loss', 'adversarial_loss']]
                epoch_df = epoch_df[['epoch', 'generator_loss', 'discriminator_loss', 'adversarial_loss']]
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
                self.write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
                self.write_log(tensorboard, 'adversarial_loss', np.mean(adv_losses), epoch)

                model_dir = os.path.join(self.save_dir, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if self.epochs >= 40:
                    if epoch % 20 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                    # self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
                    # self.adversarial_model.save_weights(os.path.join(model_dir, 'adversarial_epoch_{}.hdf5'.format(epoch)))
                else:
                    if epoch % 5 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
        
        data.close()

if __name__ == '__main__':
    gan = SR_UnetGAN()
    gan.train()