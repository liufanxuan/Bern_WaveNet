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

import pywt
import pywt.data

aaa=np.array([[ -1.66666698,   0.        ,   0.        , 299.16671753],
       [  0.        ,  -1.66666698,   0.        , 299.16671753],
       [  0.        ,   0.        ,   2.88599992, 569.75793457],
       [  0.        ,   0.        ,   0.        ,   1.        ]])
max_img=612363.6 #class one
# max_img=1172342.5 #Siemens

class Evaluate():
	def __init__(self):
		self.domain = 'valid'
		self.target_shape = [360,360,673]
		self.wave_shape = [8,176,176,337]  


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
	def generate_result_3d(self,generator,epoch,save_dir):
		if 1:
			if 1:
				if 1:
					data_path = '/media/data/fanxuan/data/PART2_h5data/data_50_wave_3d_16.h5'
					data_path_reduce = '/media/data/fanxuan/data/PART2/50_dose'
					data_path_normal = '/media/data/fanxuan/data/PART2/Normal'
					
					save_path = os.path.join(save_dir, 'valid_all')
					if not os.path.exists(save_path):
						os.makedirs(save_path)

					data = h5py.File(data_path, 'r')
					
					filenames = np.array(data[self.domain+'_filenames']).flatten()
					filenames = [x.decode('utf-8') for x in filenames]
					patients = sorted(set(x.split('+')[1] for x in filenames))
					shape3 = 16

					count2 = 0
					patient_list, mse_list, nrmse_list, mape_list, psnr_list, ssim_list = [],[],[],[],[],[]
					ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list = [],[],[],[],[]
					patient_num = 0 
					for patient in patients:
						volume = np.zeros(shape=self.wave_shape)
						gd_volume = np.zeros(shape=self.wave_shape)
						x_volume = np.zeros(shape=self.wave_shape)
						ind_volume = np.zeros(shape=self.wave_shape)                     
						count = 0                       
						print(len(filenames))
						for i in range(len(filenames)):
							if patient==filenames[i].split('+')[1]:
								count2+=1
								print('{}: {}/{}'.format(filenames[i], count2, len(filenames)))
								n = int(filenames[i].split('+')[-2])
								s = int(filenames[i].split('+')[-1])
								x_slice = np.array(data.get(self.domain+'_reduce')[i])                                       
								gd_slice = np.array(data.get(self.domain+'_normal')[i])
								x1_slice = np.expand_dims(x_slice, axis=0)
								# x_slice = np.expand_dims(x_slice, axis=0)
								out_slice = generator.predict(x1_slice)[0,:,:,:]
								if count*(shape3//4)+shape3 >= x_volume.shape[3]:	
									x_volume[:,:,:,x_volume.shape[3]-shape3:x_volume.shape[3]] += x_slice
									gd_volume[:,:,:,gd_volume.shape[3]-shape3:gd_volume.shape[3]] += gd_slice
									volume[:,:,:,volume.shape[3]-shape3:volume.shape[3]] += out_slice
									ind_volume[:,:,:,ind_volume.shape[3]-shape3:ind_volume.shape[3]] += 1
									break
								x_volume[:,:,:,count*(shape3//4):count*(shape3//4)+shape3] += x_slice
								gd_volume[:,:,:,count*(shape3//4):count*(shape3//4)+shape3] += gd_slice
								volume[:,:,:,count*(shape3//4):count*(shape3//4)+shape3] += out_slice
								ind_volume[:,:,:,count*(shape3//4):count*(shape3//4)+shape3] += 1      
								count += 1
						x_volume = x_volume/ind_volume
						print(x_volume.shape)  
						gd_volume = gd_volume/ind_volume
						volume = volume/ind_volume
						coe = {'aaa':x_volume[0,:,:,:],'aad':x_volume[1,:,:,:],'ada':x_volume[2,:,:,:],'add':x_volume[3,:,:,:],'daa':x_volume[4,:,:,:],'dad':x_volume[5,:,:,:],'dda':x_volume[6,:,:,:],'ddd':x_volume[7,:,:,:]} 
						re = pywt.idwtn(coe, 'haar', mode='symmetric', axes=None)
						x_re = re[:,:,:673]*max_img
						coe = {'aaa':gd_volume[0,:,:,:],'aad':gd_volume[1,:,:,:],'ada':gd_volume[2,:,:,:],'add':gd_volume[3,:,:,:],'daa':gd_volume[4,:,:,:],'dad':gd_volume[5,:,:,:],'dda':gd_volume[6,:,:,:],'ddd':gd_volume[7,:,:,:]} 
						re = pywt.idwtn(coe, 'haar', mode='symmetric', axes=None)
						gd_re = re[:,:,:673]*max_img 
						coe = {'aaa':volume[0,:,:,:],'aad':volume[1,:,:,:],'ada':volume[2,:,:,:],'add':volume[3,:,:,:],'daa':volume[4,:,:,:],'dad':volume[5,:,:,:],'dda':volume[6,:,:,:],'ddd':volume[7,:,:,:]} 
						re = pywt.idwtn(coe, 'haar', mode='symmetric', axes=None)
						out_re = re[:,:,:673]*max_img                
						PET_path = os.path.join(data_path_reduce,"patient_"+ str(n)+".nii")
						x_normal = np.array(nib.load(PET_path).dataobj)  
						x_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],x_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
						print(x_re.shape)
						x_volume = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],x_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)
            
						out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)
						print(out_re.shape)
						volume = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1)                                        
                                                   
						gd_path = os.path.join(data_path_normal,"patient_"+ str(n)+".nii")
						gd_normal = np.array(nib.load(gd_path).dataobj)  
						gd_re = np.concatenate((gd_normal[0:(gd_normal.shape[0]//2-176), (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)],gd_re,gd_normal[(gd_normal.shape[0]//2+176):, (gd_normal.shape[1]//2-176):(gd_normal.shape[1]//2+176)]), axis=0)  
						gd_volume = np.concatenate((gd_normal[:, :(gd_normal.shape[1]//2-176)],gd_re,gd_normal[:, (gd_normal.shape[1]//2+176):]), axis = 1)  
						patient_save_path = save_path
						if not os.path.exists(patient_save_path):
							os.makedirs(patient_save_path)
						if epoch == 200:
							ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(gd_volume, x_volume)
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
                          
                          
                          
				
					df = pd.DataFrame({ 'Patient_num': patient_list, 'MSE': mse_list,
				                    	   'NRMSE %': nrmse_list,'mape': mape_list, 'PSNR': psnr_list, 'SSIM': ssim_list})
					df = df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(mse_list),
				                		'NRMSE %': np.mean(nrmse_list), 'mape': np.mean(mape_list),
                            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)}, ignore_index=True)
					df = df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
					if epoch == 200:
						ori_df = pd.DataFrame({'Patient_num': patient_list, 'MSE': ori_mse_list,
				                    	   'NRMSE %': ori_nrmse_list, 'mape': ori_mape_list,
                                 'PSNR': ori_psnr_list, 'SSIM': ori_ssim_list})
						ori_df = ori_df.append({'Patient_num': 'Mean Value', 'MSE': np.mean(ori_mse_list),
				                		'NRMSE %': np.mean(ori_nrmse_list), 'mape': np.mean(ori_mape_list),
                            'PSNR': np.mean(ori_psnr_list), 'SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
						ori_df = ori_df[['Patient_num', 'MSE', 'NRMSE %','mape', 'PSNR', 'SSIM']]
						ori_df.to_csv(os.path.join(patient_save_path,'ori.csv'), index=False)
					df.to_csv(os.path.join(patient_save_path, 'gen_'+str(epoch)+'.csv'), index=False)

									
					data.close()
					# data2.close()   


	
	

