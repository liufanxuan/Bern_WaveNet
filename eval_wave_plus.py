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
os.environ["CUDA_VISIBLE_DEVICES"]="2" 
import pywt
import pywt.data


 # max_img = 612363.6000000001 
# min_img = 0.0 
max_ll = 1114619.1216900002 
min_ll = 0.0 
max_lh = 192557.671652 
min_lh = -225969.58185200003
max_hl = 174515.240144 
min_hl = -176200.5453 
max_hh = 47016.231371999995 
min_hh = -44065.89666000002
# img_range = max_img - min_img
ll_range = max_ll - min_ll
lh_range = max_lh - min_lh
hl_range = max_hl - min_hl
hh_range = max_hh - min_hh


aaa=np.array([[ -1.66666698,   0.        ,   0.        , 299.16671753],
       [  0.        ,  -1.66666698,   0.        , 299.16671753],
       [  0.        ,   0.        ,   2.88599992, 569.75793457],
       [  0.        ,   0.        ,   0.        ,   1.        ]])
max_img=498879.6

class Evaluate():
	def __init__(self):
		self.domain = 'valid'
		self.target_shape = [360,360,673]



	def compute_metrics_old(self, real, pred):
		mse = np.mean(np.square(real-pred))
		nrmse = np.sqrt(np.sum(np.square(real-pred))/np.sum(np.square(real)))
		psnr = 10*np.log10(np.square(1.0)/mse)
		ssim = measure.compare_ssim(real, pred)
		return mse, nrmse, psnr, ssim
   
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
   
	def generate_result_3channel(self):
		if 1:
			if 1:
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_3channel.h5'
				data_path2 = '/media/data/fanxuan/data/data_LL.h5'
				data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_10_normal.h5'
				save_dir = '/media/data/fanxuan/result/FCN_wave_3ch_VAL'
				model_path = os.path.join(save_dir, 'model/generator_epoch_190.hdf5')
				save_dir2 = '/media/data/fanxuan/result/FCN_LL_VAL'				
				model_path2 = os.path.join(save_dir2, 'model/generator_epoch_190.hdf5')
				save_path = os.path.join(save_dir, 'valid_190')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator = unet_model_2d((3,176,176))
					generator.load_weights(model_path)
					generator2 = unet_model_2d((1,176,176))
					generator2.load_weights(model_path2)
					data = h5py.File(data_path, 'r')
					data2 = h5py.File(data_path2, 'r')
					data_normal = h5py.File(data_path_normal, 'r')
					filenames = np.array(data[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					patients = sorted(set(x.split('+')[1] for x in filenames))

					count = 0
					patient_list, mse_list, nrmse_list, mape_list, psnr_list, ssim_list = [],[],[],[],[],[]
					ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list = [],[],[],[],[]
					for patient in patients:
						volume = np.zeros(shape=self.target_shape)
						gd_volume = np.zeros(shape=self.target_shape)
						x_volume = np.zeros(shape=self.target_shape)
						print(len(filenames))
						for i in range(len(filenames)):
							if patient==filenames[i].split('+')[1]:
								count += 1
								print('{}: {}/{}'.format(filenames[i], count, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x_slice = np.array(data.get(self.domain+'_reduce')[i])
								x_slice2 = np.array(data2.get(self.domain+'_DRF_10')[i])
								x_normal = np.array(data_normal.get(self.domain+'_reduce')[i])                                       
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								gd_slice2 = np.array(data2.get(self.domain+'_dose')[i])
								gd_normal = np.array(data_normal.get(self.domain+'_normal')[i])						        
								coeffs_x = x_slice2, (x_slice[0], x_slice[1], x_slice[2])
								x_re = pywt.idwt2(coeffs_x, 'haar')

								x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(x_re.shape)
								x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
								x_volume[:,:,s] += x_re * max_img
								coeffs_gd = gd_slice2, (gd_slice[0], gd_slice[1], gd_slice[2])
								gd_re = pywt.idwt2(coeffs_gd, 'haar')
								gd_re = np.concatenate((gd_normal[0:(gd_normal.shape[0]//2-176), (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)],gd_re,gd_normal[(gd_normal.shape[0]//2+176):, (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)]), axis=0)
								print(gd_re.shape)
								gd_re = np.concatenate((gd_normal[:, :(gd_normal.shape[1]//2-176)],gd_re,gd_normal[:, (gd_normal.shape[1]//2+176):]), axis = 1)                                              
								gd_volume[:,:,s] += gd_re * max_img
								# ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)
								print(np.mean(x_slice))
								x_slice = np.expand_dims(x_slice, axis=0)
								# x_slice = np.expand_dims(x_slice, axis=0)
								out_slice = generator.predict(x_slice)[0,:,:,:]
								x_slice2 = np.expand_dims(x_slice2, axis=0)
								x_slice2 = np.expand_dims(x_slice2, axis=0)
								out_slice2 = generator2.predict(x_slice2)[0,0,:,:]
								print(np.mean(x_slice2))
								print(out_slice.shape)
								coeffs_out = out_slice2, (out_slice[0], out_slice[1], out_slice[2])
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(out_re.shape)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume[:,:,s] += out_re * max_img
								
						
						patient_save_path = os.path.join(save_path, 'patients')
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						nib.save(nib.Nifti1Image(x_volume, affine=aaa), os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_reduce.nii'.format(n)))
						ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
						print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
						print('mse:{}, nrmse:{}, mape:{}, psnr:{}, ssim:{}'.format(mse, nrmse, mape, psnr, ssim))
						patient_list.append(patient)
						mse_list.append(mse)
						nrmse_list.append(nrmse * 1e2)
						mape_list.append(mape)
						psnr_list.append(psnr)
						ssim_list.append(ssim)
						ori_mse_list.append(ori_mse)
						ori_nrmse_list.append(ori_nrmse * 1e2)
						ori_mape_list.append(ori_mape)
						ori_psnr_list.append(ori_psnr)
						ori_ssim_list.append(ori_ssim) 
					df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
					df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
					df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					ori_df = pd.DataFrame({'Patient_num': patient_list, 'MSE': ori_mse_list,
				                    	   'NRMSE %': ori_nrmse_list, 'mape': ori_mape_list,
                                 'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
					ori_df = ori_df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(ori_mse_list),
				                		'NRMSE %': np.mean(ori_nrmse_list), 'mape': np.mean(ori_mape_list),
                            'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
					ori_df = ori_df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					df.to_csv(os.path.join(patient_save_path, 'gen.csv'), index=False)
					ori_df.to_csv(os.path.join(patient_save_path,'ori.csv'), index=False)
					
					data.close()
	def generate_result_4channel(self):
		if 1:
			if 1:
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_4ch_range0_1.h5'
				data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_10_normal.h5'
				save_dir = '/media/data/fanxuan/result/FCN_4ch_range0_1'
				model_path = os.path.join(save_dir, 'model/generator_epoch_150.hdf5') 
				save_path = os.path.join(save_dir, 'valid_150')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator = unet_model_2d((4,176,176))
					generator.load_weights(model_path)

					data = h5py.File(data_path, 'r')
					data_normal = h5py.File(data_path_normal, 'r')
					filenames = np.array(data[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					patients = sorted(set(x.split('+')[1] for x in filenames))

					count = 0
					patient_list, mse_list, nrmse_list, mape_list, psnr_list, ssim_list = [],[],[],[],[],[]
					ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list = [],[],[],[],[]
					for patient in patients:
						volume = np.zeros(shape=self.target_shape)
						gd_volume = np.zeros(shape=self.target_shape)
						x_volume = np.zeros(shape=self.target_shape)
						print(len(filenames))
						for i in range(len(filenames)):
							if patient==filenames[i].split('+')[1]:
								count += 1
								print('{}: {}/{}'.format(filenames[i], count, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x_slice = np.array(data.get(self.domain+'_reduce')[i])
								x_normal = np.array(data_normal.get(self.domain+'_reduce')[i])*max_img
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								gd_normal = np.array(data_normal.get(self.domain+'_normal')[i])*max_img						        
								coeffs_x = x_slice[0]*ll_range, (x_slice[1]*lh_range+min_lh, x_slice[2]*hl_range+min_hl, x_slice[3]*hh_range+min_hh)
								x_re = pywt.idwt2(coeffs_x, 'haar')

								x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
								mse = np.mean(np.square(np.array(x_re) - np.array(x_normal)), axis=-1)
								print(np.mean(mse))                                                   
								x_volume[:,:,s] += x_re
								coeffs_gd= gd_slice[0]*ll_range, (gd_slice[1]*lh_range+min_lh, gd_slice[2]*hl_range+min_hl, gd_slice[3]*hh_range+min_hh)
								                         
								gd_re = pywt.idwt2(coeffs_gd, 'haar')
								gd_re = np.concatenate((gd_normal[0:(gd_normal.shape[0]//2-176), (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)],gd_re,gd_normal[(gd_normal.shape[0]//2+176):, (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)]), axis=0)
								gd_re = np.concatenate((gd_normal[:, :(gd_normal.shape[1]//2-176)],gd_re,gd_normal[:, (gd_normal.shape[1]//2+176):]), axis = 1)                                              
								gd_volume[:,:,s] += gd_re
								mse = np.mean(np.square(np.array(gd_re) - np.array(gd_normal)), axis=-1)
								print(np.mean(mse))                                         
								# ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)
								mse = np.mean(np.square(np.array(x_slice) - np.array(np.array(data.get(self.domain+'_reduce')[i]))), axis=-1)
								print('change:{}'.format(np.mean(mse)))
								x_slice = np.expand_dims(x_slice, axis=0)
								# x_slice = np.expand_dims(x_slice, axis=0)
								out_slice = generator.predict(x_slice)[0,:,:,:]
								coeffs_out = out_slice[0]*ll_range, (out_slice[1]*lh_range+min_lh, out_slice[2]*hl_range+min_hl, out_slice[3]*hh_range+min_hh)
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume[:,:,s] += out_re 
								
						
						patient_save_path = os.path.join(save_path, 'patients')
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						nib.save(nib.Nifti1Image(x_volume, affine=aaa), os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_reduce.nii'.format(n)))
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
						ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
						
						print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
						print('mse:{}, nrmse:{}, mape:{}, psnr:{}, ssim:{}'.format(mse, nrmse, mape, psnr, ssim))
						patient_list.append(patient)
						mse_list.append(mse)
						nrmse_list.append(nrmse * 1e2)
						mape_list.append(mape)
						psnr_list.append(psnr)
						ssim_list.append(ssim)
						ori_mse_list.append(ori_mse)
						ori_nrmse_list.append(ori_nrmse * 1e2)
						ori_mape_list.append(ori_mape)
						ori_psnr_list.append(ori_psnr)
						ori_ssim_list.append(ori_ssim) 
					df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
					df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
					df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					ori_df = pd.DataFrame({'Patient_num': patient_list, 'MSE': ori_mse_list,
				                    	   'NRMSE %': ori_nrmse_list, 'mape': ori_mape_list,
                                 'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
					ori_df = ori_df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(ori_mse_list),
				                		'NRMSE %': np.mean(ori_nrmse_list), 'mape': np.mean(ori_mape_list),
                            'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
					ori_df = ori_df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					df.to_csv(os.path.join(patient_save_path, 'gen.csv'), index=False)
					ori_df.to_csv(os.path.join(patient_save_path,'ori.csv'), index=False)
					
					data.close()
					data2.close()
	
	def generate_result_wave(self):
		if 1:
			if 1:
				data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_10_normal.h5'
				data_path_LL = '/media/data/fanxuan/data/data_LL.h5'
				data_path_LH = '/media/data/fanxuan/data/data_LH.h5'
				data_path_HL = '/media/data/fanxuan/data/data_HL.h5'
				data_path_HH = '/media/data/fanxuan/data/data_HH.h5'
				save_dir = '/media/data/fanxuan/result/'
				model_path_LL = os.path.join(save_dir, 'FCN_LL_VAL/model/generator_epoch_200.hdf5') 
				model_path_LH = os.path.join(save_dir, 'FCN_LH_VAL/model/generator_epoch_150.hdf5')
				model_path_HL = os.path.join(save_dir, 'FCN_HL_VAL/model/generator_epoch_150.hdf5')
				model_path_HH = os.path.join(save_dir, 'FCN_HH_VAL/model/generator_epoch_80.hdf5')
				save_path = os.path.join(save_dir, 'FCN_LL_VAL/valid_200+150+150+80')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator_LL = unet_model_2d((1,176,176))
					generator_LL.load_weights(model_path_LL)
					generator_LH = unet_model_2d((1,176,176))
					generator_LH.load_weights(model_path_LH)
					generator_HL = unet_model_2d((1,176,176))
					generator_HL.load_weights(model_path_HL)
					generator_HH = unet_model_2d((1,176,176))
					generator_HH.load_weights(model_path_HH)
					data_LL = h5py.File(data_path_LL, 'r')
					data_LH = h5py.File(data_path_LH, 'r')
					data_HL = h5py.File(data_path_HL, 'r')
					data_HH = h5py.File(data_path_HH, 'r')
					data_normal = h5py.File(data_path_normal, 'r')
					filenames = np.array(data_LL[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					patients = sorted(set(x.split('+')[1] for x in filenames))
					patient_list, mse_list, nrmse_list, mape_list, psnr_list, ssim_list = [],[],[],[],[],[]
					ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list = [],[],[],[],[]
					count = 0
					for patient in patients:
						volume = np.zeros(shape=self.target_shape)
						gd_volume = np.zeros(shape=self.target_shape)
						x_volume = np.zeros(shape=self.target_shape)

						print(len(filenames))
						for i in range(len(filenames)):
							if patient==filenames[i].split('+')[1]:
								count += 1
								print('{}: {}/{}'.format(filenames[i], count, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x_normal = np.array(data_normal.get(self.domain+'_reduce')[i])
								x_LL = np.array(data_LL.get(self.domain+'_DRF_10')[i])
								x_LH = np.array(data_LH.get(self.domain+'_DRF_10')[i])
								x_HL = np.array(data_HL.get(self.domain+'_DRF_10')[i])
								x_HH = np.array(data_HH.get(self.domain+'_DRF_10')[i])
								coeffs2 = x_LL, (x_LH, x_HL, x_HH)
								x_slice = pywt.idwt2(coeffs2, 'haar')
								x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_slice,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(x_re.shape)
								x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
								x_volume[:,:,s] += x_re * max_img
								gd_normal = np.array(data_normal.get(self.domain+'_normal')[i])
								gd_volume[:,:,s] += gd_normal * max_img                               
				
								x_LL = np.expand_dims(x_LL, axis=0)
								x_LL = np.expand_dims(x_LL, axis=0)
								out_LL = generator_LL.predict(x_LL)[0,0,:,:]
								x_LH = np.expand_dims(x_LH, axis=0)
								x_LH = np.expand_dims(x_LH, axis=0)
								out_LH = generator_LH.predict(x_LH)[0,0,:,:]
								x_HL = np.expand_dims(x_HL, axis=0)
								x_HL = np.expand_dims(x_HL, axis=0)
								out_HL = generator_HL.predict(x_HL)[0,0,:,:]
								x_HH = np.expand_dims(x_HH, axis=0)
								x_HH = np.expand_dims(x_HH, axis=0)
								out_HH = generator_HH.predict(x_HH)[0,0,:,:]
								coeffs_out = out_LL, (out_LH, out_HL, out_HH)
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(out_re.shape)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume[:,:,s] += out_re * max_img                                                   

						patient_save_path = os.path.join(save_path, 'patients')
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						nib.save(nib.Nifti1Image(x_volume, affine=aaa), os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_reduce.nii'.format(n)))
						ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
						print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
						print('mse:{}, nrmse:{}, mape:{}, psnr:{}, ssim:{}'.format(mse, nrmse, mape, psnr, ssim))
						patient_list.append(patient)
						print(patient_list)
						mse_list.append(mse)
						nrmse_list.append(nrmse * 1e2)
						mape_list.append(mape)
						psnr_list.append(psnr)
						print(psnr_list)
						ssim_list.append(ssim)
						print(ssim_list)
						ori_mse_list.append(ori_mse)
						ori_nrmse_list.append(ori_nrmse * 1e2)
						ori_mape_list.append(ori_mape)
						ori_psnr_list.append(ori_psnr)
						ori_ssim_list.append(ori_ssim) 
					df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
					df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
					df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					ori_df = pd.DataFrame({'Patient_num': patient_list, 'MSE': ori_mse_list,
				                    	   'NRMSE %': ori_nrmse_list, 'mape': ori_mape_list,
                                 'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
					ori_df = ori_df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(ori_mse_list),
				                		'NRMSE %': np.mean(ori_nrmse_list), 'mape': np.mean(ori_mape_list),
                            'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
					ori_df = ori_df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					df.to_csv(os.path.join(patient_save_path, 'gen.csv'), index=False)
					ori_df.to_csv(os.path.join(patient_save_path,'ori.csv'), index=False)
					data_normal.close()
					data_LL.close()
					data_HL.close()
					data_LH.close()
					data_HH.close()
	def generate_result_3channel_add(self):
		if 1:
			if 1:
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_3wave_range1_1.h5'
				data_path2 = '/media/data/fanxuan/data/PART2_h5data/data_ll_range1_1.h5'
				data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_10_normal.h5'
				save_dir = '/media/data/fanxuan/result/FCN_3w_range1_1'
				model_path = os.path.join(save_dir, 'model/generator_epoch_200.hdf5')
				save_dir2 = '/media/data/fanxuan/result/FCN_LL_VAL'				
				model_path2 = os.path.join(save_dir2, 'model/generator_epoch_200.hdf5')
				save_path = os.path.join(save_dir, 'valid_200+200')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator = unet_model_2d((1,176,176))
					generator.load_weights(model_path)
					generator2 = unet_model_2d((1,176,176))
					generator2.load_weights(model_path2)
					data = h5py.File(data_path, 'r')
					data2 = h5py.File(data_path2, 'r')
					data_normal = h5py.File(data_path_normal, 'r')
					filenames = np.array(data2[self.domain+'_filenames']).flatten()
					filenames2 = np.array(data[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					filenames2 = [x.decode('utf-8') for x in filenames2]
					patients = sorted(set(x.split('+')[1] for x in filenames))

					count = 0
					patient_list, mse_list, nrmse_list, mape_list, psnr_list, ssim_list = [],[],[],[],[],[]
					ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list = [],[],[],[],[]
					for patient in patients:
						volume = np.zeros(shape=self.target_shape)
						gd_volume = np.zeros(shape=self.target_shape)
						x_volume = np.zeros(shape=self.target_shape)
						num = 0
						print(len(filenames))
						for i in range(len(filenames)):
							if patient==filenames[i].split('+')[1]:
								count += 1
								print('{}: {},{}/{}'.format(filenames[i], num,count, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x_LH = np.array(data.get(self.domain+'_reduce')[num])
								print('x_LH:{}',format(filenames2[num])) 
								gd_LH = np.array(data.get(self.domain+'_normal')[num])
								num += 1 
								x_HL = np.array(data.get(self.domain+'_reduce')[num])
								gd_HL = np.array(data.get(self.domain+'_normal')[num])
								print('x_HL:{}',format(filenames2[num])) 								
								num += 1 
								x_HH = np.array(data.get(self.domain+'_reduce')[num])
								gd_HH = np.array(data.get(self.domain+'_normal')[num])
								print('x_HH:{}',format(filenames2[num])) 								
								num += 1                                                     
								x_slice2 = np.array(data2.get(self.domain+'_reduce')[i])
								x_normal = np.array(data_normal.get(self.domain+'_reduce')[i]) * max_img                                     
								
								gd_slice2 = np.array(data2.get(self.domain+'_normal')[i])
								gd_normal = np.array(data_normal.get(self.domain+'_normal')[i]) * max_img						        
								coeffs_x = x_slice2*ll_range, (x_LH*lh_range/2, x_HL*hl_range/2, x_HH*hh_range/2)
								x_re = pywt.idwt2(coeffs_x, 'haar')
                
								x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								
								x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
								print(np.mean(np.square(x_re-x_normal)))
								x_volume[:,:,s] += x_re
                                       
								coeffs_gd = gd_slice2*ll_range, (gd_LH*lh_range/2, gd_HL*hl_range/2, gd_HH*hh_range/2)
								gd_re = pywt.idwt2(coeffs_gd, 'haar')
								gd_re = np.concatenate((gd_normal[0:(gd_normal.shape[0]//2-176), (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)],gd_re,gd_normal[(gd_normal.shape[0]//2+176):, (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)]), axis=0)

								gd_re = np.concatenate((gd_normal[:, :(gd_normal.shape[1]//2-176)],gd_re,gd_normal[:, (gd_normal.shape[1]//2+176):]), axis = 1)    
								print(np.mean(np.square(gd_re-gd_normal)))                                          
								gd_volume[:,:,s] += gd_re
								# ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)

								x_LH = np.expand_dims(x_LH, axis=0)
								x_LH = np.expand_dims(x_LH, axis=0)
								out_LH= generator.predict(x_LH)[0,0,:,:]
								x_HL = np.expand_dims(x_HL, axis=0)
								x_HL = np.expand_dims(x_HL, axis=0)
								out_HL = generator.predict(x_HL)[0,0,:,:]
								x_HH = np.expand_dims(x_HH, axis=0)
								x_HH = np.expand_dims(x_HH, axis=0)
								out_HH = generator.predict(x_HH)[0,0,:,:]
								x_slice2 = np.expand_dims(x_slice2, axis=0)
								x_slice2 = np.expand_dims(x_slice2, axis=0)
								out_slice2 = generator2.predict(x_slice2)[0,0,:,:]
                                                                                
								coeffs_out = out_slice2*ll_range, (out_LH*lh_range/2, out_HL*hl_range/2, out_HH*hh_range/2)
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(out_re.shape)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume[:,:,s] += out_re
								
						
						patient_save_path = os.path.join(save_path, 'patients')
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						nib.save(nib.Nifti1Image(x_volume, affine=aaa), os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_reduce.nii'.format(n)))
						ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
						print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
						print('mse:{}, nrmse:{}, mape:{}, psnr:{}, ssim:{}'.format(mse, nrmse, mape, psnr, ssim))
						patient_list.append(patient)
						mse_list.append(mse)
						nrmse_list.append(nrmse * 1e2)
						mape_list.append(mape)
						psnr_list.append(psnr)
						ssim_list.append(ssim)
						ori_mse_list.append(ori_mse)
						ori_nrmse_list.append(ori_nrmse * 1e2)
						ori_mape_list.append(ori_mape)
						ori_psnr_list.append(ori_psnr)
						ori_ssim_list.append(ori_ssim) 
					df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
					df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
					df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					ori_df = pd.DataFrame({'Patient_num': patient_list, 'MSE': ori_mse_list,
				                    	   'NRMSE %': ori_nrmse_list, 'mape': ori_mape_list,
                                 'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
					ori_df = ori_df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(ori_mse_list),
				                		'NRMSE %': np.mean(ori_nrmse_list), 'mape': np.mean(ori_mape_list),
                            'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
					ori_df = ori_df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					df.to_csv(os.path.join(patient_save_path, 'gen.csv'), index=False)
					ori_df.to_csv(os.path.join(patient_save_path,'ori.csv'), index=False)
					
					data.close()
					data2.close()
eva = Evaluate()
eva.generate_result_4channel()