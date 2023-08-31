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

from Unet import unet_model_2d
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import pywt
import pywt.data

aaa=np.array([[ -1.66666698,   0.        ,   0.        , 299.16671753],
       [  0.        ,  -1.66666698,   0.        , 299.16671753],
       [  0.        ,   0.        ,   2.88599992, 569.75793457],
       [  0.        ,   0.        ,   0.        ,   1.        ]])
max_img=612363.6
#max_img=1172342.5
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
   
	def generate_result(self):
		if 1:
			if 1:
				#data_path = '/media/data/fanxuan/data/PART2_Siemens_h5data/data_Siemens_normal_50.h5'
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_normal_10.h5'
				save_dir = '/media/data/fanxuan/result/FCN_10_nomal'
				model_path = os.path.join(save_dir, 'model/generator_epoch_250.hdf5') 
				save_path = os.path.join(save_dir, 'valid_250')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator = unet_model_2d((1,360,360))
					generator.load_weights(model_path)

					data = h5py.File(data_path, 'r')
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
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								x_volume[:,:,s] += x_slice * max_img
								# ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)
								x_slice = np.expand_dims(x_slice, axis=0)
								x_slice = np.expand_dims(x_slice, axis=0)
								out_slice = generator.predict(x_slice)[0,0,:,:]
								volume[:,:,s] += out_slice * max_img
								gd_volume[:,:,s] += gd_slice * max_img
								
						volume[volume<0] = 0
						patient_save_path = save_path
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						'''
            nib.save(nib.Nifti1Image(volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						nib.save(nib.Nifti1Image(x_volume, affine=aaa), os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_reduce.nii'.format(n)))
                                                            '''
						#ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
						#print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
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
	def generate_u_result(self):
		if 1:
			if 1:
				#data_path = '/media/data/fanxuan/data/PART2_Siemens_h5data/data_Siemens_normal_50.h5'
				data_path = '/media/data/fanxuan/data/PART2_h5data/data_normal_10.h5'
				data_path2 = '/media/data/fanxuan/data/PART2_h5data/data_wave_10_haar.h5'
				save_dir = '/media/data/fanxuan/result/FCN_10_nomal'
				save_dir2 = '/media/data/fanxuan/result/FCN_10_wave_haar'
				model_path = os.path.join(save_dir, 'model/generator_epoch_200.hdf5') 
				model_path2 = os.path.join(save_dir2, 'model/generator_epoch_200.hdf5') 
				save_path = os.path.join(save_dir, 'valid_200/two')
				wavelet = 'haar'
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator = unet_model_2d((1,360,360))
					generator.load_weights(model_path)
					generator2 = unet_model_2d((4,176,176))
					generator2.load_weights(model_path2)

					data = h5py.File(data_path, 'r')
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
								x_normal = np.array(data.get(self.domain+'_reduce')[i])
								x_slice = np.array(data.get(self.domain+'_reduce')[i])
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								x_volume[:,:,s] += x_slice * max_img
								# ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)
								x_slice = np.expand_dims(x_slice, axis=0)
								x_slice = np.expand_dims(x_slice, axis=0)
								PET1 = generator.predict(x_slice)[0,0,:,:]
								PET1 = PET1[(PET1.shape[0]//2-176):(PET1.shape[0]//2+176), (PET1.shape[1]//2-176):(PET1.shape[1]//2+176)]
								coeffs_pet = pywt.dwt2(PET1, wavelet)
								LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
								PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])  
								
								out_slice = generator2.predict(np.expand_dims(PET_wave, axis=0))[0,:,:,:]
								print(np.mean(x_slice))
								print(out_slice.shape)
								coeffs_out = out_slice[0], (out_slice[1], out_slice[2], out_slice[3])
								out_re = pywt.idwt2(coeffs_out, 'haar')
								out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
								print(out_re.shape)
								out_re = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                                
								volume[:,:,s] += out_re * max_img                                                      
                                                                           
								gd_volume[:,:,s] += gd_slice * max_img
								
						volume[volume<0] = 0
						patient_save_path = save_path
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						'''
            nib.save(nib.Nifti1Image(volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						nib.save(nib.Nifti1Image(x_volume, affine=aaa), os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_reduce.nii'.format(n)))
                                                            '''
						#ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
						mse, nrmse, mape, psnr, ssim = self.compute_metrics(gd_volume, volume)
						#print('ori_mse:{}, ori_nrmse:{}, ori_mape:{}, ori_psnr:{}, ori_ssim:{}'.format(ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim))
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
	def generate_all_result(self):
		if 1:
			#data_path = '/media/data/fanxuan/data/PART2_Siemens_h5data/data_Siemens_normal_50.h5'
			data_path = '/media/data/fanxuan/data/PART2_h5data/data_normal_10.h5'
			save_dir = '/media/data/fanxuan/result/FCN_10_nomal'
			save_path = os.path.join(save_dir, 'valid_all')
			start_num = 15
			end_num = 40
      
			model_list, ave_mse_list, ave_nrmse_list, ave_mape_list, ave_psnr_list, ave_ssim_list = [],[],[],[],[],[]
			for model_index in range(start_num,end_num+1):
				model_num = model_index*10
				model_path = os.path.join(save_dir, 'model/generator_epoch_'+str(model_num)+'.hdf5')

				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator = unet_model_2d((1,360,360))
					generator.load_weights(model_path)

					data = h5py.File(data_path, 'r')
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
								print('{}:{}: {}/{}'.format(model_num, filenames[i], count, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x_slice = np.array(data.get(self.domain+'_reduce')[i])
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								x_volume[:,:,s] += x_slice * max_img
								# ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)
								x_slice = np.expand_dims(x_slice, axis=0)
								x_slice = np.expand_dims(x_slice, axis=0)
								out_slice = generator.predict(x_slice)[0,0,:,:]
								volume[:,:,s] += out_slice * max_img
								gd_volume[:,:,s] += gd_slice * max_img
								
						volume[volume<0] = 0
						patient_save_path = save_path
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
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
					ave_mse_list.append(np.mean(mse_list)) 
					ave_nrmse_list.append(np.mean(nrmse_list))
					ave_mape_list.append(np.mean(mape_list)) 
					ave_psnr_list.append(np.mean(psnr_list))
					ave_ssim_list.append(np.mean(ssim_list))
					
					data.close()
			df_all = pd.DataFrame({ 'Model_epoch_num': model_list, 'MSE': ave_mse_list,'NRMSE %': ave_nrmse_list,'mape': ave_mape_list, 'PSNR': ave_psnr_list, 'SSIM': ave_ssim_list})
			df_all = df_all[['Model_epoch_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
			df_all.to_csv(os.path.join(patient_save_path, 'all_from'+str(start_num)+'to'+str(end_num)+'.csv'), index=False)
	def generate_result_wave(self):
		if 1:
			if 1:
				data_path_LL = '/content/drive/MyDrive/Bern/uExplorerPART2_h5data/data_LL.h5'
				data_path_LH = '/content/drive/MyDrive/Bern/uExplorerPART2_h5data/data_LH.h5'
				data_path_HL = '/content/drive/MyDrive/Bern/uExplorerPART2_h5data/data_HL.h5'
				data_path_HH = '/content/drive/MyDrive/Bern/uExplorerPART2_h5data/data_HH.h5'
				save_dir = '/content/drive/MyDrive/Bern/result/FCN_2D'
				model_path_LL = os.path.join(save_dir, 'model/LL/generator_epoch_100.hdf5') 
				model_path_LH = os.path.join(save_dir, 'model/LH/generator_epoch_100.hdf5')
				model_path_HL = os.path.join(save_dir, 'model/HL/generator_epoch_100.hdf5')
				model_path_HH = os.path.join(save_dir, 'model/HH/generator_epoch_100.hdf5')
				save_path = os.path.join(save_dir, 'valid2')
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
					sess.run(tf.global_variables_initializer())
					K.set_session(sess)
					generator_LL = SR_UnetGAN().build_generator()
					generator_LL.load_weights(model_path_LL)
					generator_LH = SR_UnetGAN().build_generator()
					generator_LH.load_weights(model_path_LH)
					generator_HL = SR_UnetGAN().build_generator()
					generator_HL.load_weights(model_path_HL)
					generator_HH = SR_UnetGAN().build_generator()
					generator_HH.load_weights(model_path_HH)
					data_LL = h5py.File(data_path_LL, 'r')
					data_LH = h5py.File(data_path_LH, 'r')
					data_HL = h5py.File(data_path_HL, 'r')
					data_HH = h5py.File(data_path_HH, 'r')
					filenames = np.array(data_LL[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					patients = sorted(set(x.split('+')[1] for x in filenames))

					count = 0
					for patient in patients:
						volume = np.zeros(shape=self.target_shape)
						gd_volume = np.zeros(shape=self.target_shape)
						slice_list, mse_list, nrmse_list, psnr_list, ssim_list = [],[],[],[],[]
						ori_mse_list, ori_nrmse_list, ori_psnr_list, ori_ssim_list = [],[],[],[]
						print(len(filenames))
						for i in range(len(filenames)):
							if patient==filenames[i].split('+')[1]:
								count += 1
								print('{}: {}/{}'.format(filenames[i], count, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x_LL = np.array(data_LL.get(self.domain+'_DRF_10')[i])
								x_LH = np.array(data_LH.get(self.domain+'_DRF_10')[i])
								x_HL = np.array(data_HL.get(self.domain+'_DRF_10')[i])
								x_HH = np.array(data_HH.get(self.domain+'_DRF_10')[i])
								coeffs2 = x_LL, (x_LH, x_HL, x_HH)
								x_slice = pywt.idwt2(coeffs2, 'haar')
				
								gd_LL = np.array(data_LL.get(self.domain+'_dose')[i])
								gd_LH = np.array(data_LH.get(self.domain+'_dose')[i])
								gd_HL = np.array(data_HL.get(self.domain+'_dose')[i])
								gd_HH = np.array(data_HH.get(self.domain+'_dose')[i])
								coeffs_gd = gd_LL, (gd_LH, gd_HL, gd_HH)
								gd_slice = pywt.idwt2(coeffs_gd, 'haar')
				
								ori_mse, ori_nrmse, ori_psnr, ori_ssim = self.compute_metrics(gd_slice, x_slice)
				
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
								out_slice = pywt.idwt2(coeffs_out, 'haar')
								volume[:,:,s] += out_slice * max_img
								gd_volume[:,:,s] += gd_slice * max_img
								mse, nrmse, psnr, ssim = self.compute_metrics(out_slice, gd_slice)
								slice_list.append(s)
								mse_list.append(mse * 1e5)
								nrmse_list.append(nrmse * 1e2)
								psnr_list.append(psnr)
								ssim_list.append(ssim)
								ori_mse_list.append(ori_mse * 1e5)
								ori_nrmse_list.append(ori_nrmse * 1e2)
								ori_psnr_list.append(ori_psnr)
								ori_ssim_list.append(ori_ssim)
						
						patient_save_path = os.path.join(save_path, 'patients')
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(volume, affine=aaa),
												 os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_generated.nii'.format(n)))
						nib.save(nib.Nifti1Image(gd_volume, affine=aaa),
												 os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_source_{}_ori.nii'.format(n)))
						df = pd.DataFrame({'Slice_num': slice_list, 'MSE (*e-05)': mse_list,
				                    	   'NRMSE %': nrmse_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
						df = df.append({'Slice_num': 'Mean Value', 'MSE (*e-05)': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
						df = df[['Slice_num', 'MSE (*e-05)', 'NRMSE %', 'PSNR', 'SSIM']]
						ori_df = pd.DataFrame({'Slice_num': slice_list, 'MSE (*e-05)': ori_mse_list,
				                    	   'NRMSE %': ori_nrmse_list, 'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
						ori_df = ori_df.append({'Slice_num': 'Mean Value', 'MSE (*e-05)': np.mean(ori_mse_list),
				                		'NRMSE %': np.mean(ori_nrmse_list), 'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
						ori_df = ori_df[['Slice_num', 'MSE (*e-05)', 'NRMSE %', 'PSNR', 'SSIM']]
						df.to_csv(os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_gen.csv'), index=False)
						ori_df.to_csv(os.path.join(patient_save_path, os.path.splitext(patient)[0]+'_ori.csv'), index=False)
					
					data_LL.close()
					data_HL.close()
					data_LH.close()
					data_HH.close()
eva = Evaluate()
eva.generate_u_result()