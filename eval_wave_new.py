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
from MWUnet import unet_mw_2d_2
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import pywt
import pywt.data

aaa=np.array([[ -1.66666698,   0.        ,   0.        , 299.16671753],
       [  0.        ,  -1.66666698,   0.        , 299.16671753],
       [  0.        ,   0.        ,   2.88599992, 569.75793457],
       [  0.        ,   0.        ,   0.        ,   1.        ]])
max_img=612363.6
# max_img=1172342.5

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
	def generate_result_4channel(self):
		if 1:
			if 1:
				#data_path = '/media/data/fanxuan/data/PART2_Siemens_h5data/data_Siemens_50.h5'
				#data_path_normal = '/media/data/fanxuan/data/PART2_Siemens_h5data/data_Siemens_normal_50.h5'
				#data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_10_wave.h5'
				#data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_50_new.h5'
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_wave_10_haar.h5'
				data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_normal_10.h5'
				save_dir = '/media/data/fanxuan/result//FCN_10_wave_mw_haar/2.0'
				model_path = os.path.join(save_dir, 'model/generator_epoch_300.hdf5') 
				save_path = os.path.join(save_dir, 'valid_300')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator = unet_mw_2d_2((4,176,176))
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
								#x_slice = np.array(data.get(self.domain+'_reduce_50')[i])
								x_slice = np.array(data.get(self.domain+'_reduce')[i])
								x_normal = np.array(data_normal.get(self.domain+'_reduce')[i])                                       
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								gd_normal = np.array(data_normal.get(self.domain+'_normal')[i])						        
								coeffs_x = x_slice[0], (x_slice[1], x_slice[2], x_slice[3])
								x_re = pywt.idwt2(coeffs_x, 'haar')

								x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(x_re.shape)
								x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
								x_volume[:,:,s] += x_re * max_img 
								coeffs_gd = gd_slice[0], (gd_slice[1], gd_slice[2], gd_slice[3])
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
								print(np.mean(x_slice))
								print(out_slice.shape)
								coeffs_out = out_slice[0], (out_slice[1], out_slice[2], out_slice[3])
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(out_re.shape)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume[:,:,s] += out_re * max_img
								
								
						
						patient_save_path = os.path.join(save_path, 'patients')
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						'''nib.save(nib.Nifti1Image(volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						nib.save(nib.Nifti1Image(x_volume, affine=aaa), os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_reduce.nii'.format(n)))
                                                            '''
						volume[volume < 0] = 0
						x_volume[x_volume < 0] = 0
						gd_volume[gd_volume < 0] = 0
						#ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
						'''print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))'''
						print('mse:{}, nrmse:{}, mape:{}, psnr:{}, ssim:{}'.format(mse, nrmse, mape, psnr, ssim))
						patient_list.append(patient)
						mse_list.append(mse)
						nrmse_list.append(nrmse * 1e2)
						mape_list.append(mape)
						psnr_list.append(psnr)
						ssim_list.append(ssim)
						'''ori_mse_list.append(ori_mse)
						ori_nrmse_list.append(ori_nrmse * 1e2)
						ori_mape_list.append(ori_mape)
						ori_psnr_list.append(ori_psnr)
						ori_ssim_list.append(ori_ssim) 
					ori_df = pd.DataFrame({'Patient_num': patient_list, 'MSE': ori_mse_list,
				                    	   'NRMSE %': ori_nrmse_list, 'mape': ori_mape_list,
                                 'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
					ori_df = ori_df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(ori_mse_list),
				                		'NRMSE %': np.mean(ori_nrmse_list), 'mape': np.mean(ori_mape_list),
                            'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
					ori_df = ori_df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					ori_df.to_csv(os.path.join(patient_save_path,'ori.csv'), index=False)'''
					df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
					df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
					df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]

					df.to_csv(os.path.join(patient_save_path, 'gen.csv'), index=False)
				
					data.close()
					data_normal.close()   
	def generate_all_result_4channel(self):
		if 1:
			if 1:
				#data_path = '/media/data/fanxuan/data/PART2_Siemens_h5data/data_Siemens_50.h5'
				#data_path_normal = '/media/data/fanxuan/data/PART2_Siemens_h5data/data_Siemens_normal_50.h5'
				#data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_10_wave.h5'
				#data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_50_new.h5'
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_wave_10_bio3_7.h5'
				data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_normal_10.h5'
				save_dir = '/media/data/fanxuan/result/FCN_10_wave_bior3_7'
				wave_name = 'bior3.7'
				len_wave = 169
				start_num = 15
				end_num = 40
				save_path = os.path.join(save_dir, 'valid_all')
				if not os.path.exists(save_path):
					os.makedirs(save_path)
      
      
				model_list, ave_mse_list, ave_nrmse_list, ave_mape_list, ave_psnr_list, ave_ssim_list = [],[],[],[],[],[]
				for model_index in range(start_num,end_num+1):
					model_num = model_index*10
					model_path = os.path.join(save_dir, 'model/generator_epoch_'+str(model_num)+'.hdf5')
  				#save_path = os.path.join(save_dir, 'valid_300')

  
					config = tf.ConfigProto()
					config.gpu_options.allow_growth = True
					with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
						sess.run(tf.global_variables_initializer())
						K.set_session(sess)
						generator = unet_mw_2d_2((176,176,4))
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
							#print(len(filenames))
							for i in range(len(filenames)):
								if patient==filenames[i].split('+')[1]:
									count += 1
									print('{}:{}: {}/{}'.format(model_num, filenames[i], count, len(filenames)))
									n = int(filenames[i].split('+')[-2])
									s = int(filenames[i].split('+')[-1])
									x_slice = np.array(data.get(self.domain+'_reduce')[i])
									x_normal = np.array(data_normal.get(self.domain+'_reduce')[i])                                       
									gd_slice = np.array(data.get(self.domain+'_normal')[i])
									gd_normal = np.array(data_normal.get(self.domain+'_normal')[i])						        
									coeffs_x = x_slice[0], (x_slice[1], x_slice[2], x_slice[3])
									x_re = pywt.idwt2(coeffs_x, wave_name)
									x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-len_wave), (x_normal.shape[1]//2-len_wave):(x_normal.shape[1]//2+len_wave)],x_re,x_normal[(x_normal.shape[0]//2+len_wave):, (x_normal.shape[1]//2-len_wave):(x_normal.shape[1]//2+len_wave)]), axis=0)
									x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-len_wave)],x_re,x_normal[:, (x_normal.shape[1]//2+len_wave):]), axis = 1)
									x_volume[:,:,s] += x_re * max_img 
									coeffs_gd = gd_slice[0], (gd_slice[1], gd_slice[2], gd_slice[3])
									gd_re = pywt.idwt2(coeffs_gd, wave_name)
									gd_re = np.concatenate((gd_normal[0:(gd_normal.shape[0]//2-len_wave), (gd_normal.shape[1]//2-len_wave):(gd_normal.shape[1]//2+len_wave)],gd_re,gd_normal[(gd_normal.shape[0]//2+len_wave):, (gd_normal.shape[1]//2-len_wave):(gd_normal.shape[1]//2+len_wave)]), axis=0)
									gd_re = np.concatenate((gd_normal[:, :(gd_normal.shape[1]//2-len_wave)],gd_re,gd_normal[:, (gd_normal.shape[1]//2+len_wave):]), axis = 1)                                             
									gd_volume[:,:,s] += gd_re * max_img
									x_slice = np.expand_dims(x_slice, axis=0)
									out_slice = generator.predict(x_slice)[0,:,:,:]
									coeffs_out = out_slice[0], (out_slice[1], out_slice[2], out_slice[3])
									out_re = pywt.idwt2(coeffs_out, wave_name)
									out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-len_wave), (x_normal.shape[1]//2-len_wave):(x_normal.shape[1]//2+len_wave)],out_re,x_normal[(x_normal.shape[0]//2+len_wave):, (x_normal.shape[1]//2-len_wave):(x_normal.shape[1]//2+len_wave)]), axis=0)
									out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-len_wave)],out_re,x_normal[:, (x_normal.shape[1]//2+len_wave):]), axis = 1)                                                
									volume[:,:,s] += out_re * max_img
	
							patient_save_path = save_path
							if not os.path.exists(patient_save_path):
								os.makedirs(patient_save_path)
							volume[volume < 0] = 0
							x_volume[x_volume < 0] = 0
							gd_volume[gd_volume < 0] = 0
							if model_index == start_num:
								ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
								print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
								ori_mse_list.append(ori_mse)
								ori_nrmse_list.append(ori_nrmse * 1e2)
								ori_mape_list.append(ori_mape)
								ori_psnr_list.append(ori_psnr)
								ori_ssim_list.append(ori_ssim) 
							mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
							print('mse:{}, nrmse:{}, mape:{}, psnr:{}, ssim:{}'.format(mse, nrmse, mape, psnr, ssim))
							patient_list.append(patient)
							mse_list.append(mse)
							nrmse_list.append(nrmse * 1e2)
							mape_list.append(mape)
							psnr_list.append(psnr)
							ssim_list.append(ssim)
  						
						if model_index == start_num:
							ori_df = pd.DataFrame({'Patient_num': patient_list, 'MSE': ori_mse_list,
    				                    	   'NRMSE %': ori_nrmse_list, 'mape': ori_mape_list,
                                     'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
							ori_df = ori_df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(ori_mse_list),
    				                		'NRMSE %': np.mean(ori_nrmse_list), 'mape': np.mean(ori_mape_list),
                                'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
							ori_df = ori_df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
							ori_df.to_csv(os.path.join(patient_save_path,'ori.csv'), index=False)
							model_list.append(0) 
							ave_mse_list.append(np.mean(ori_mse_list)) 
							ave_nrmse_list.append(np.mean(ori_nrmse_list))
							ave_mape_list.append(np.mean(ori_mape_list)) 
							ave_psnr_list.append(np.mean(ori_psnr_list))
							ave_ssim_list.append(np.mean(ori_ssim_list))
						df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
  				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
						df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
  				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                              'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
						df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
  
						df.to_csv(os.path.join(patient_save_path, 'gen_'+str(model_num)+'.csv'), index=False)
						model_list.append(model_num) 
						ave_mse_list.append(np.mean(mse_list)) 
						ave_nrmse_list.append(np.mean(nrmse_list))
						ave_mape_list.append(np.mean(mape_list)) 
						ave_psnr_list.append(np.mean(psnr_list))
						ave_ssim_list.append(np.mean(ssim_list))
						data.close()
						data_normal.close() 
				df_all = pd.DataFrame({ 'Model_epoch_num': model_list, 'MSE': ave_mse_list,'NRMSE %': ave_nrmse_list,'mape': ave_mape_list, 'PSNR': ave_psnr_list, 'SSIM': ave_ssim_list})
				df_all = df_all[['Model_epoch_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
				df_all.to_csv(os.path.join(patient_save_path, 'all_from'+str(start_num)+'to'+str(end_num)+'.csv'), index=False)
				  

	def generate_result_4channel_multi(self):
		if 1:
			if 1:
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_10_wave.h5'
				data_path_normal = '/media/data/fanxuan/data/PART2_h5data/data_50_new.h5'
				save_dir = '/media/data/fanxuan/result/FCN_10_50_wave_new'
				model1_path = os.path.join(save_dir, 'model/generator1_epoch_100.hdf5')
				model2_path = os.path.join(save_dir, 'model/generator2_epoch_100.hdf5') 
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
					data_normal = h5py.File(data_path_normal, 'r')
					filenames = np.array(data[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					patients = sorted(set(x.split('+')[1] for x in filenames))

					count = 0
					patient_list, mse_list, nrmse_list, mape_list, psnr_list, ssim_list = [],[],[],[],[],[]
					ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list = [],[],[],[],[]
					mse_10_list, nrmse_10_list, mape_10_list, psnr_10_list, ssim_10_list = [],[],[],[],[]
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
								x_normal = np.array(data_normal.get(self.domain+'_reduce')[i])                                       
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								gd_normal = np.array(data_normal.get(self.domain+'_normal')[i])		
                				        
								coeffs_x = x1_slice[0], (x1_slice[1], x1_slice[2], x1_slice[3])
								x_re = pywt.idwt2(coeffs_x, 'haar')
								x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(x_re.shape)
								x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
								x1_volume[:,:,s] += x_re * max_img
                                         
								coeffs_x = x2_slice[0], (x2_slice[1], x2_slice[2], x2_slice[3])
								x_re = pywt.idwt2(coeffs_x, 'haar')
								x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(x_re.shape)
								x_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
								x2_volume[:,:,s] += x_re * max_img                                        
                                         
								coeffs_gd = gd_slice[0], (gd_slice[1], gd_slice[2], gd_slice[3])
								gd_re = pywt.idwt2(coeffs_gd, 'haar')
								gd_re = np.concatenate((gd_normal[0:(gd_normal.shape[0]//2-176), (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)],gd_re,gd_normal[(gd_normal.shape[0]//2+176):, (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)]), axis=0)
								print(gd_re.shape)
								gd_re = np.concatenate((gd_normal[:, :(gd_normal.shape[1]//2-176)],gd_re,gd_normal[:, (gd_normal.shape[1]//2+176):]), axis = 1)                                              
								gd_volume[:,:,s] += gd_re * max_img
								# ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)
								# print(np.mean(x1_slice))
								x1_slice = np.expand_dims(x1_slice, axis=0)
								# x_slice = np.expand_dims(x_slice, axis=0)
								out1_slice = generator1.predict(x1_slice)
								con_slice =  np.concatenate((out1_slice,x1_slice), axis=1)
								out2_slice = generator2.predict(con_slice)[0,:,:,:]
								print(out2_slice.shape)
								coeffs_out = out2_slice[0], (out2_slice[1], out2_slice[2], out2_slice[3])
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(out_re.shape)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume2[:,:,s] += out_re * max_img
                                         
								out1_slice = out1_slice[0,:,:,:]                    
								coeffs_out = out1_slice[0], (out1_slice[1], out1_slice[2], out1_slice[3])
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(out_re.shape)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume1[:,:,s] += out_re * max_img
								
						
						patient_save_path = os.path.join(save_path, 'patients')
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(volume1, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated10.nii'.format(n)))
						nib.save(nib.Nifti1Image(volume2, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						
						ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x1_volume)
						mse_10, nrmse_10, mape_10, psnr_10, ssim_10 = self.compute_metrics(x2_volume, volume1)
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume2)
						print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
						print('mse:{}, nrmse:{}, mape:{}, psnr:{}, ssim:{}'.format(mse, nrmse, mape, psnr, ssim))
						patient_list.append(patient)
						mse_list.append(mse)
						nrmse_list.append(nrmse * 1e2)
						mape_list.append(mape)
						psnr_list.append(psnr)
						ssim_list.append(ssim)
						mse_10_list.append(mse_10)
						nrmse_10_list.append(nrmse_10 * 1e2)
						mape_10_list.append(mape_10)
						psnr_10_list.append(psnr_10)
						ssim_10_list.append(ssim_10)
						ori_mse_list.append(ori_mse)
						ori_nrmse_list.append(ori_nrmse * 1e2)
						ori_mape_list.append(ori_mape)
						ori_psnr_list.append(ori_psnr)
						ori_ssim_list.append(ori_ssim) 
					df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list, 
                                 'MSE_10': mse_10_list, 'NRMSE_10 %': nrmse_10_list,'mape_10': mape_10_list, 
                                 'PSNR_10': psnr_10_list, 'SSIM_10': ssim_10_list})
					df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list),
                            'MSE_10': np.mean(mse_10_list),
				                		'NRMSE_10 %': np.mean(nrmse_10_list), 'mape_10': np.mean(mape_10_list),
                            'PSNR_10': np.mean(psnr_10_list), 'SSIM_10': np.mean(ssim_10_list)}, ignore_index=True)
					df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM', 'MSE_10', 'NRMSE_10 %','mape_10', 'PSNR_10', 'SSIM_10']]
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
	
	

eva = Evaluate()
eva.generate_all_result_4channel()