
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
import random
import shutil
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
import random
from FCN_3D import SR_UnetGAN
#from Unet3d import unet_se_3d
from WMUnet_3d import unet_model_3d
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
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
    #path = '/media/data/fanxuan/data/TEST1/test/'
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
        #print(re_1.shape)
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
    return volume

def write_test_3d(normal_path_list):
		save_dir = '/media/data/fanxuan/result/FCN_multi_3_10/'
		#model_path = os.path.join(save_dir, 'model/generator_epoch_109.hdf5') 
		save_path = os.path.join(save_dir, 'test_19')
		if not os.path.exists(save_path):
				os.makedirs(save_path)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
				sess.run(tf.global_variables_initializer())
				K.set_session(sess)
				generator = unet_model_3d((8,176,176,16))
				generator.load_weights('/media/data/fanxuan/result/FCN_multi_3_10/model/generator_epoch_143.hdf5')
				start = 0
				shape3 = 16
				#file_base = '/media/data/fanxuan/data/'
				file_base = '/media/data/fanxuan/Challenge_first_round/'
				for niifile in normal_path_list:
						file_name = niifile
						#file_base = '/media/data/fanxuan/data/'
						print(file_name)
						#PET_1_path = os.path.join(file_base,'test',file_name)
						PET_1_path = os.path.join('/media/data/fanxuan/data/TEST2/test/',file_name)
						for niifile in os.listdir(PET_1_path):
								PET_1_path = os.path.join(PET_1_path,niifile)
						if 'nii.gz' not in PET_1_path:
								print('False!!!not os.path.exists test!!!!')
								continue
						PET1,ori_data,ori_aaa = data_reshape(PET_1_path)
						print(PET1.shape)       
						coe_pet1 = trans_wave(PET1)
						print(coe_pet1.shape)  
						coe_pet1 = coe_pet1*10
            
						#gt_path = os.path.join(file_base,'normal',file_name,'GD.nii.gz')
						gt_path = os.path.join(file_base,'ground-truth', file_name+'.nii.gz')
						if not os.path.exists(gt_path):
								print('False!!!not os.path.exists normal!!!!')
								continue
############test
						gt,gt_data,gt_aaa = data_reshape(gt_path)
						coe_gt = trans_wave(gt)
						coe_gt = coe_gt*10

						pet_volume = np.zeros(shape=coe_pet1.shape)
						gt_volume = np.zeros(shape=coe_pet1.shape)
#####################
						j = 0
						volume = np.zeros(shape=coe_pet1.shape)
						ind_volume = np.zeros(shape=coe_pet1.shape)
						while j < coe_pet1.shape[3]:
								#print(j)
								nextstep = shape3//4
								if j+shape3+1 < coe_pet1.shape[3]:
										if np.mean(coe_pet1[:,:,:,j:j+shape3])<30/max_img:
												nextstep = shape3//4*3
								else:
										#ori_slice = coe_pet1[:,:,:,coe_pet1.shape[3]-shape3:coe_pet1.shape[3]]
										org_tran = np.expand_dims(coe_pet1[:,:,:,coe_pet1.shape[3]-shape3:coe_pet1.shape[3]], axis=0)
										print(org_tran.shape)
										out_slice = generator.predict(org_tran)[0,:,:,:]
###########################test
										gt_tran = coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]]
										gt_volume[:,:,:,gt_volume.shape[3]-shape3:gt_volume.shape[3]] += gt_tran
										pet_tran = coe_pet1[:,:,:,coe_pet1.shape[3]-shape3:coe_pet1.shape[3]]
										pet_volume[:,:,:,pet_volume.shape[3]-shape3:pet_volume.shape[3]] += pet_tran
										val_loss = np.mean(np.square(np.array(gt_tran) - np.array(out_slice)))
										ori_loss = np.mean(np.square(np.array(gt_tran) - np.array(pet_tran)))
										print("    val_loss: {}\n".format(val_loss))
										print("    ori_loss: {}\n".format(ori_loss))
###############################
										volume[:,:,:,volume.shape[3]-shape3:volume.shape[3]] += out_slice
										ind_volume[:,:,:,ind_volume.shape[3]-shape3:ind_volume.shape[3]] += 1
										break

								org_tran = np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0)
								out_slice = generator.predict(org_tran)[0,:,:,:]
								volume[:,:,:,j:j+shape3] += out_slice
								ind_volume[:,:,:,j:j+shape3] += 1
###########################test
								gt_tran = coe_gt[:,:,:,j:j+shape3]
								gt_volume[:,:,:,j:j+shape3] += gt_tran
								pet_tran = coe_pet1[:,:,:,j:j+shape3]
								pet_volume[:,:,:,j:j+shape3] += pet_tran
								print("ori_loss: {}\n".format(np.mean(np.square(np.array(pet_tran) - np.array(gt_tran)))))
								print("gen_loss: {}\n".format(np.mean(np.square(np.array(out_slice) - np.array(gt_tran)))))
##################################
								j = j+nextstep
            #x_volume = x_volume/ind_volume
 
						#gd_volume = gd_volume/ind_volume
						volume = volume/ind_volume*0.1
						print(volume.shape) 
						coe = {'aaa':volume[0,:,:,:],'aad':volume[1,:,:,:],'ada':volume[2,:,:,:],'add':volume[3,:,:,:],'daa':volume[4,:,:,:],'dad':volume[5,:,:,:],'dda':volume[6,:,:,:],'ddd':volume[7,:,:,:]} 
						coe = coe
						re = pywt.idwtn(coe, 'haar', mode='symmetric', axes=None)
						out_re = unreshape(re,ori_data) 
						ran = random.random()
						out_re[out_re<0] = ran
						out_re = out_re
						patient_save_path = os.path.join(file_base,'gen_143',file_name)
						if not os.path.exists(patient_save_path):
								os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(out_re, affine=ori_aaa),os.path.join(patient_save_path, 'gen.nii.gz'))

def write_test_3d_100(normal_path_list,model_num,dose):
		#save_dir = '/media/data/fanxuan/result/FCN_multi_3_10/'
		#model_path = os.path.join(save_dir, 'model/generator_epoch_109.hdf5') 
		#save_path = os.path.join(save_dir, 'test_19')
		#if not os.path.exists(save_path):
				#os.makedirs(save_path)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
				sess.run(tf.global_variables_initializer())
				K.set_session(sess)
				generator = unet_model_3d((8,176,176,16))
				generator.load_weights("/media/data/fanxuan/result/FCN_multi_10_drf_10_6E-5/model/generator_epoch_"+model_num+".hdf5")
				start = 0
				shape3 = 16
				file_base = '/media/data/fanxuan/data/test_all'
				#file_base = '/media/data/fanxuan/Challenge_first_round/'
				for niifile in normal_path_list:
						file_name = niifile
						file_base = '/media/data/fanxuan/data/'
						print(file_name)
						PET_1_path = os.path.join(file_base,'test',file_name)
						#PET_1_path = os.path.join('/media/data/fanxuan/data/TEST2/test/',file_name)
						for niifile in os.listdir(PET_1_path):
								PET_1_path = os.path.join(PET_1_path,niifile)
						if 'nii.gz' not in PET_1_path:
								print('False!!!not os.path.exists test!!!!')
								continue
						PET1,ori_data,ori_aaa = data_reshape(PET_1_path)
						print(PET1.shape)       
						coe_pet1 = trans_wave(PET1)
						print(coe_pet1.shape)  
						coe_pet1 = coe_pet1*10
            
						gt_path = os.path.join(file_base,'normal',file_name,'GD.nii.gz')
						#gt_path = os.path.join(file_base,'ground-truth', file_name+'.nii.gz')
						if not os.path.exists(gt_path):
								print('False!!!not os.path.exists normal!!!!')
								continue
############test
						gt,gt_data,gt_aaa = data_reshape(gt_path)
						coe_gt = trans_wave(gt)
						coe_gt = coe_gt*10

						pet_volume = np.zeros(shape=coe_pet1.shape)
						gt_volume = np.zeros(shape=coe_pet1.shape)
#####################
						j = 0
						volume = np.zeros(shape=coe_pet1.shape)
						ind_volume = np.zeros(shape=coe_pet1.shape)
						while j < coe_pet1.shape[3]:
								#print(j)
								nextstep = shape3//4
								if j+shape3+1 < coe_pet1.shape[3]:
										if np.mean(coe_pet1[:,:,:,j:j+shape3])<30/max_img:
												nextstep = shape3//4*3
								else:
										#ori_slice = coe_pet1[:,:,:,coe_pet1.shape[3]-shape3:coe_pet1.shape[3]]
										org_tran = np.expand_dims(coe_pet1[:,:,:,coe_pet1.shape[3]-shape3:coe_pet1.shape[3]], axis=0)
										print(org_tran.shape)
										out_slice = generator.predict(org_tran)[0,:,:,:]
###########################test
										gt_tran = coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]]
										gt_volume[:,:,:,gt_volume.shape[3]-shape3:gt_volume.shape[3]] += gt_tran
										pet_tran = coe_pet1[:,:,:,coe_pet1.shape[3]-shape3:coe_pet1.shape[3]]
										pet_volume[:,:,:,pet_volume.shape[3]-shape3:pet_volume.shape[3]] += pet_tran
										val_loss = np.mean(np.square(np.array(gt_tran) - np.array(out_slice)))
										ori_loss = np.mean(np.square(np.array(gt_tran) - np.array(pet_tran)))
										print("    val_loss: {}\n".format(val_loss))
										print("    ori_loss: {}\n".format(ori_loss))
###############################
										volume[:,:,:,volume.shape[3]-shape3:volume.shape[3]] += out_slice
										ind_volume[:,:,:,ind_volume.shape[3]-shape3:ind_volume.shape[3]] += 1
										break

								org_tran = np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0)
								out_slice = generator.predict(org_tran)[0,:,:,:]
								volume[:,:,:,j:j+shape3] += out_slice
								ind_volume[:,:,:,j:j+shape3] += 1
###########################test
								gt_tran = coe_gt[:,:,:,j:j+shape3]
								gt_volume[:,:,:,j:j+shape3] += gt_tran
								pet_tran = coe_pet1[:,:,:,j:j+shape3]
								pet_volume[:,:,:,j:j+shape3] += pet_tran
								print("ori_loss: {}\n".format(np.mean(np.square(np.array(pet_tran) - np.array(gt_tran)))))
								print("gen_loss: {}\n".format(np.mean(np.square(np.array(out_slice) - np.array(gt_tran)))))
##################################
								j = j+nextstep
            #x_volume = x_volume/ind_volume
 
						#gd_volume = gd_volume/ind_volume
						volume = volume/ind_volume*0.1
						print(volume.shape) 
						coe = {'aaa':volume[0,:,:,:],'aad':volume[1,:,:,:],'ada':volume[2,:,:,:],'add':volume[3,:,:,:],'daa':volume[4,:,:,:],'dad':volume[5,:,:,:],'dda':volume[6,:,:,:],'ddd':volume[7,:,:,:]} 
						coe = coe
						re = pywt.idwtn(coe, 'haar', mode='symmetric', axes=None)
						out_re = unreshape(re,ori_data) 
						ran = random.random()
						out_re[out_re<0] = ran
						out_re = out_re
						patient_save_path = os.path.join(file_base,'test_final','gen_10_epoch'+dose+model_num,file_name)
						if not os.path.exists(patient_save_path):
								os.makedirs(patient_save_path)
						nib.save(nib.Nifti1Image(out_re, affine=ori_aaa),os.path.join(patient_save_path, 'gen.nii'))


df = pd.read_csv("/home/xuan/code/drf_all.csv")
#test_name_list = df['drf_4']
normal_path_list = find_filepath()
#random.shuffle(normal_path_list)
#test_list = normal_path_list	
#write_test_3d(normal_path_list)	
'''for model_num in range(180,200):
		print('model_num***********************************************')
		print(model_num)
		write_test_3d_100(test_name_list,str(model_num))	
		#write_test_3d(normal_path_list)'''
#write_test_3d_100(test_name_list,str(12),'drf_4')
#write_test_3d_100(test_name_list,str(13))
#test_name_list = df['drf_20']	
#write_test_3d_100(test_name_list,str(12),'drf_20')
test_name_list = df['drf_10']	
write_test_3d_100(test_name_list,str(6),'drf_10')
'''

eva = Evaluate()
eva.generate_result_3d(test_list)'''