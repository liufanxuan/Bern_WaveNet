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
from Unet import unet_model_2d
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import pywt
import pywt.data

aaa=np.array([[ -1.66666698,   0.        ,   0.        , 299.16671753],
       [  0.        ,  -1.66666698,   0.        , 299.16671753],
       [  0.        ,   0.        ,   2.88599992, 569.75793457],
       [  0.        ,   0.        ,   0.        ,   1.        ]])
max_img=612363.6

class Evaluate():
	def __init__(self):
		self.domain = 'valid'
		self.target_shape = [360,360,673]


	def compute_metrics(self, real_input, pred_input):
		real = real_input.copy()
		real[real<1] = 0
		pred = pred_input.copy()
		pred[pred<1] = 0
		mse = np.mean(np.square(real-pred))
		print(mse)
		nrmse = np.sqrt(mse) / (np.max(real)-np.min(real))
		print('nrmse:{}'.format(nrmse))
		ok_idx = np.where(real!=0)
		mape = np.mean(np.abs((real[ok_idx] - pred[ok_idx]) / real[ok_idx]))
		print('mape:{}'.format(mape))
		PIXEL_MAX = np.max(real)
		psnr = 20*np.log10(PIXEL_MAX / np.sqrt(np.mean(np.square(real-pred))))
		print('psnr:{}'.format(psnr))
		real_norm = real / float(np.max(real))
		pred_norm = pred / float(np.max(pred))
		ssim = measure.compare_ssim(real_norm, pred_norm)

		return mse, nrmse, mape, psnr, ssim


 

	def generate_result_4channel_multi(self):
		if 1:
			if 1:
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_10_wave.h5'
				save_dir = '/media/data/fanxuan/result/FCN_10_50_wave_new'
				model1_path = os.path.join(save_dir, 'model/generator1_epoch_200.hdf5')
				model2_path = os.path.join(save_dir, 'model/generator2_epoch_200.hdf5') 
				save_path = os.path.join(save_dir, 'valid_100')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator1 = unet_model_2d((4,176,176))
					generator1.load_weights(model1_path)
					generator2 = unet_model_2d((8,176,176))
					generator2.load_weights(model2_path)

					data = h5py.File(data_path, 'r')
					filenames = np.array(data[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					patients = sorted(set(x.split('+')[1] for x in filenames))

					count = 0
					patient_list, mse_list, nrmse_list, mape_list, psnr_list, ssim_list = [],[],[],[],[],[]
					ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list = [],[],[],[],[]
					mse_10_list, nrmse_10_list, mape_10_list, psnr_10_list, ssim_10_list = [],[],[],[],[]
					mse_wave = []
					for patient in patients:
						volume1 = np.zeros(shape=self.target_shape)
						volume2 = np.zeros(shape=self.target_shape)
						gd_volume = np.zeros(shape=self.target_shape)
						x1_volume = np.zeros(shape=self.target_shape)
						x2_volume = np.zeros(shape=self.target_shape)
						print(len(filenames))
						for i in range(len(filenames)):
							if patient==filenames[i].split('+')[1]:
								count += 1
								print('{}: {}/{}'.format(filenames[i], count, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x1_slice = np.array(data.get(self.domain+'_reduce_50')[i])
								x2_slice = np.array(data.get(self.domain+'_reduce_10')[i])                                     
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								x1_slice = np.expand_dims(x1_slice, axis=0)
								out1_slice = generator1.predict(x1_slice)
								con_slice =  np.concatenate((out1_slice,x1_slice), axis=1)
								out2_slice = generator2.predict(con_slice)[0,:,:,:]
								a2 = np.mean(np.square(np.array(gd_slice) - np.array(out2_slice)), axis=-1)
								mse_wave.append(np.mean(a2))        
								print(np.mean(a2))
								

					print('mse:{}'.format(np.mean(mse_wave)))
						
					
					data.close()
	
	

eva = Evaluate()
eva.generate_result_4channel_multi()