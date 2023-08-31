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

import dicom2nifti
import random
import shutil
import h5py
import tables
import nibabel as nib
from sklearn.model_selection import KFold
import logging
import time
import SimpleITK as sitk
import pydicom
import gzip
import scipy.ndimage

from FCN_3D import SR_UnetGAN
#from Unet3d import unet_se_3d
from WMUnet_3d import unet_model_3d
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import pywt
import pywt.data

'''aaa=np.array([[ -1.66666698,   0.        ,   0.        , 299.16671753],
       [  0.        ,  -1.66666698,   0.        , 299.16671753],
       [  0.        ,   0.        ,   2.88599992, 569.75793457],
       [  0.        ,   0.        ,   0.        ,   1.        ]])'''
# max_img=612363.6
# max_img=1172342.5
max_img = 10312549.44
def trans_wave(PET):
    coe = pywt.dwtn(PET, 'haar', mode='symmetric', axes=None)
    coe_pet = np.array([coe['aaa'],coe['aad'],coe['ada'],coe['add'],coe['daa'],coe['dad'],coe['dda'],coe['ddd']])
    return coe_pet
def find_filepath():
    filepath_list = []
    path = '/media/data/fanxuan/data/test/'
    for niifile in os.listdir(path):
        if '.' not in niifile:
            filepath_list.append(niifile)
    return filepath_list
def data_reshape(PET_path):
    nii_data = nib.load(PET_path)
    ori_aaa = nii_data.affine
    print(PET_path)
    header = nii_data.header
    PET = np.array(nii_data.dataobj)
    if PET.shape[0] == 360:
        PET2 = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
    else:
        PET_1 = PET[:352,:352]
        PET_2 = PET[PET.shape[0]-352:,:352]
        PET_3 = PET[:352,PET.shape[0]-352:]
        PET_4 = PET[PET.shape[0]-352:,PET.shape[0]-352:]
        PET2 = np.concatenate((PET_1,PET_2,PET_3,PET_4),axis=2)
    '''if PET2.shape[2]%2 != 0: 
        PET2 = PET2 [:,:,1:]'''
    print(PET2.shape)
    return PET2/max_img,PET,ori_aaa
def unreshape(re,PET):
    #nii_data = nib.load(PET_path)
    #print(PET_path)
    #header = nii_data.header
    #PET = np.array(nii_data.dataobj)
    re = re*max_img
    print(PET.shape)
    if PET.shape[0] == 360:
        out_re = re[:,:,:PET.shape[2]]
        PET1 = PET
        PET1 = PET1[(PET1.shape[0]//2-180):(PET1.shape[0]//2+180), (PET1.shape[1]//2-180):(PET1.shape[1]//2+180)] 
        x_normal = PET1
        #PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
        out_re = np.concatenate((x_normal[0:(x_normal.shape[0]//2-176), (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)],out_re,x_normal[(x_normal.shape[0]//2+176):, (x_normal.shape[1]//2-176):(x_normal.shape[1]//2+176)]), axis=0)

        volume = np.concatenate((x_normal[:, :(x_normal.shape[1]//2-176)],out_re,x_normal[:, (x_normal.shape[1]//2+176):]), axis = 1) 
        print(volume.shape) 
    else:
        volume = np.zeros(shape=PET.shape)
        ind = np.zeros(shape=PET.shape)
        re_1 = re[:,:,:re.shape[2]//4]
        print(re_1.shape)
        re_2 = re[:,:,re.shape[2]//4:re.shape[2]//2]
        re_3 = re[:,:,re.shape[2]//2:re.shape[2]//4*3]
        re_4 = re[:,:,re.shape[2]//4*3:]
        volume[:352,:352,:] += re_1
        ind[:352,:352,:] += 1
        volume[PET.shape[0]-352:,:352,:] += re_2
        ind[PET.shape[0]-352:,:352,:] += 1
        volume[:352,PET.shape[0]-352:,:] += re_3
        ind[:352,PET.shape[0]-352:,:] += 1
        volume[PET.shape[0]-352:,PET.shape[0]-352:,:] += re_4
        ind[PET.shape[0]-352:,PET.shape[0]-352:,:] += 1
        #print(ind)
        volume = volume/ind
        print(volume.shape)    
    return volume

def write_test_3d(normal_path_list):
		if 1:
				start = 0
				shape3 = 16
				file_base = '/media/data/fanxuan/data/'
				for niifile in normal_path_list:
						file_name = niifile
            #file_base = '/media/data/fanxuan/data/'
						print(file_name)
						PET_1_path = os.path.join(file_base,'gen',file_name)
						for niifile in os.listdir(PET_1_path):
								PET_1_path = os.path.join(PET_1_path,niifile)
						if 'nii.gz' not in PET_1_path:
								print('False!!!not os.path.exists test!!!!')
								continue
						PET1,ori_data,ori_aaa = data_reshape(PET_1_path)
						coe_pet1 = trans_wave(PET1)
						#coe = pywt.dwtn(PET1, 'haar', mode='symmetric', axes=None)       
						gt_path = os.path.join(file_base,'normal',file_name,'GD.nii.gz')
						if not os.path.exists(gt_path):
								print('False!!!not os.path.exists normal!!!!')
								continue
						gt,gt_data,gt_aaa = data_reshape(gt_path)
						coe_gt = trans_wave(gt)

						j = 0
						volume = np.zeros(shape=coe_pet1.shape)
						#gd_volume = np.zeros(shape=coe_pet1.shape)
						#x_volume = np.zeros(shape=coe_pet1.shape)
						ind_volume = np.zeros(shape=coe_pet1.shape)
						while j < coe_pet1.shape[3]:
								#print(j)
								nextstep = shape3//4
								if j+shape3+1 < coe_pet1.shape[3]:
										if np.mean(coe_pet1[:,:,:,j:j+shape3])<30/max_img:
												nextstep = shape3//4*3
								else:
										org_tran = np.expand_dims(coe_gt[:,:,:,coe_pet1.shape[3]-shape3:coe_pet1.shape[3]], axis=0)
										out_slice = org_tran[0,:,:,:]
                    #gt_tran = np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0)
                    #gd_volume[:,:,:,gd_volume.shape[3]-shape3:gd_volume.shape[3]] += gt_tran
										volume[:,:,:,volume.shape[3]-shape3:volume.shape[3]] += out_slice
										ind_volume[:,:,:,ind_volume.shape[3]-shape3:ind_volume.shape[3]] += 1
										break
								

								org_tran = np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0)
								out_slice = org_tran[0,:,:,:]
								volume[:,:,:,j:j+shape3] += out_slice
								ind_volume[:,:,:,j:j+shape3] += 1
                #gt_tran = np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0)
								j = j+nextstep
            #x_volume = x_volume/ind_volume
 
						#gd_volume = gd_volume/ind_volume
						volume = volume/ind_volume

						#volume = coe_pet1
						print(volume.shape) 
						coe = {'aaa':volume[0,:,:,:],'aad':volume[1,:,:,:],'ada':volume[2,:,:,:],'add':volume[3,:,:,:],'daa':volume[4,:,:,:],'dad':volume[5,:,:,:],'dda':volume[6,:,:,:],'ddd':volume[7,:,:,:]} 
						re = pywt.idwtn(coe, 'haar', mode='symmetric', axes=None)
						out_re = unreshape(re,ori_data)
						print(np.mean(np.square(out_re-ori_data)))
						'''patient_save_path = os.path.join(file_base,'gen',file_name)
						if not os.path.exists(patient_save_path):
								os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(volume, affine=ori_aaa),os.path.join(patient_save_path, 'gen.nii.gz'))'''



normal_path_list = find_filepath()
#test_list = normal_path_list	
write_test_3d(normal_path_list)	
'''
eva = Evaluate()
eva.generate_result_3d(test_list)'''