import os
import numpy as np
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
import pandas as pd

import pywt
import pywt.data

max_img = 10312549.44

def find_filepath():
    filepath_list = []
    path = '/media/data/fanxuan/data/data_all_nii/Subject/'
    for root, dirs, files in os.walk(path):
        if 'Normal' in root:
            for niifile in files:
                filepath_list.append(os.path.join(root,niifile))
    return filepath_list
def find_filepath_test():
    filepath_list = []
    path = '/media/data/fanxuan/data/test/'
    for niifile in os.listdir(path):
        if '.' not in niifile:
            filepath_list.append(niifile)
    return filepath_list
    
def find_filepath_old(dose):
    filepath_list = []
    path = '/media/data/fanxuan/data/data_all_nii'
    for root, dirs, files in os.walk(path):
        if dose in root:
            for niifile in files:
                filepath_list.append(os.path.join(root,niifile))
    return filepath_list
    
def compress_delete_file(data_file):

    gz_path = data_file+'.gz'
    if os.path.exists(gz_path):
        os.remove(gz_path)
    if '.gz' not in data_file:
        pf = open(gz_path, 'wb')
        data = open(data_file,'rb').read()
        data_comp = gzip.compress(data)
        pf.write(data_comp)
        pf.close()

        if os.path.exists(data_file):
            os.remove(data_file)
        else:
            print('False!!!!!')
     
'''def data_reshape(PET_path):
    nii_data = nib.load(PET_path)
    print(PET_path)
    header = nii_data.header
    PET = np.array(nii_data.dataobj)
    if PET.shape[0] == 360:
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
    else:
        PET = scipy.ndimage.zoom(PET,176*2/PET.shape[0] , order=1)    
    return PET/max_img
'''
def data_reshape(PET_path):
    #Subject
    nii_data = nib.load(PET_path)
    print(PET_path)
    header = nii_data.header
    PET = np.array(nii_data.dataobj)
    if PET.shape[0] == 440:
        pad = np.zeros((448,448,PET.shape[2]))
        h,w,z = pad.shape
        pad[4:h-4,4:w-4] = PET
        PET = pad
    else:
        PET = scipy.ndimage.zoom(PET,224*2/PET.shape[0] , order=1)    
    return PET/max_img
    
def trans_wave(PET):
    coe = pywt.dwtn(PET, 'haar', mode='symmetric', axes=None)
    coe_pet = np.array([coe['aaa'],coe['aad'],coe['ada'],coe['add'],coe['daa'],coe['dad'],coe['dda'],coe['ddd']])
    return coe_pet

def write_h5_3d(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_2_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_2 = target_file.create_earray(target_file.root,dataset+'_4_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_3 = target_file.create_earray(target_file.root,dataset+'_10_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_4 = target_file.create_earray(target_file.root,dataset+'_20_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_5 = target_file.create_earray(target_file.root,dataset+'_50_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_6 = target_file.create_earray(target_file.root,dataset+'_100_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=10000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=10000000)



    for normal_path in normal_path_list:
        if 1:
            file_name = os.path.basename(normal_path)
            file_base = os.path.dirname(os.path.dirname(normal_path))
            print(file_name)
            PET_1_path = os.path.join(file_base,'2_dose',file_name)
            if not os.path.exists(PET_1_path):
                continue
            PET_2_path = os.path.join(file_base,'4_dose',file_name)
            if not os.path.exists(PET_2_path):
                continue
            PET_3_path = os.path.join(file_base,'10_dose',file_name)
            if not os.path.exists(PET_3_path):
                continue
            PET_4_path = os.path.join(file_base,'20_dose',file_name)
            if not os.path.exists(PET_4_path):
                continue
            PET_5_path = os.path.join(file_base,'50_dose',file_name)
            if not os.path.exists(PET_5_path):
                continue
            PET_6_path = os.path.join(file_base,'100_dose',file_name)
            if not os.path.exists(PET_6_path):
                continue
            gt_path = os.path.join(file_base,'Normal',file_name)
            if not os.path.exists(gt_path):
                continue
            PET1 = data_reshape(PET_1_path)
            PET2 = data_reshape(PET_2_path)
            PET3 = data_reshape(PET_3_path)
            PET4 = data_reshape(PET_4_path)
            PET5 = data_reshape(PET_5_path)
            PET6 = data_reshape(PET_6_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_pet2 = trans_wave(PET2)
            coe_pet3 = trans_wave(PET3)
            coe_pet4 = trans_wave(PET4)
            coe_pet5 = trans_wave(PET5) 
            coe_pet6 = trans_wave(PET6)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3] or coe_gt.shape[3] != coe_pet2.shape[3] or coe_gt.shape[3] != coe_pet3.shape[3] or coe_gt.shape[3] != coe_pet4.shape[3] or coe_gt.shape[3] != coe_pet5.shape[3] or coe_gt.shape[3] != coe_pet6.shape[3]: 
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,j:j+shape3], axis=0))
                PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,j:j+shape3], axis=0))
                PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,j:j+shape3], axis=0))
                PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,j:j+shape3], axis=0))
                PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep


            
def write_test_3d(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)


    for niifile in normal_path_list:
        if 1:
            file_name = niifile
            file_base = '/media/data/fanxuan/data/TEST2/'
            print(file_name)
            PET_1_path = os.path.join(file_base,'test',file_name)
            for niifile in os.listdir(PET_1_path):
                PET_1_path = os.path.join(PET_1_path,niifile)
            if 'nii.gz' not in PET_1_path:
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join('/media/data/fanxuan/Challenge_first_round/ground-truth/'+file_name+'.nii.gz')
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists normal!!!!')
                print(gt_path)
                continue
            PET1 = data_reshape(PET_1_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3]: 
                print('False!!!shape different!!!!')
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep

def write_test_3d_d(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)


    for niifile in normal_path_list:
        if 1:
            file_name = niifile
            file_base = '/media/data/fanxuan/data/'
            print(file_name)
            PET_1_path = os.path.join(file_base,'test',file_name)
            for niifile in os.listdir(PET_1_path):
                PET_1_path = os.path.join(PET_1_path,niifile)
            if 'nii.gz' not in PET_1_path:
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join(file_base,'normal',file_name)
            for niifile in os.listdir(gt_path):
                gt_path = os.path.join(gt_path,niifile)
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists normal!!!!')
                print(gt_path)
                continue
            PET1 = data_reshape(PET_1_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3]: 
                print('False!!!shape different!!!!')
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep

def write_divide_3d(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)

    for niifile in normal_path_list:
        if 1:
            file_name = niifile
            file_base = '/media/data/fanxuan/data/TEST2/'
            print(file_name)
            PET_1_path = os.path.join(file_base,'test',file_name)
            for niifile in os.listdir(PET_1_path):
                PET_1_path = os.path.join(PET_1_path,niifile)
            if 'nii.gz' not in PET_1_path:
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join('/media/data/fanxuan/Challenge_first_round/ground-truth/'+file_name+'.nii.gz')
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists normal!!!!')
                print(gt_path)
                continue
            PET1 = data_reshape(PET_1_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3]: 
                print('False!!!shape different!!!!')
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep

def write_64_3d(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 64
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,shape3,shape3,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,shape3,shape3,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)


    for normal_path in normal_path_list:


        if 1:
            file_name = os.path.basename(normal_path)
            file_base_data = os.path.dirname(normal_path)
            file_base = os.path.dirname(os.path.dirname(normal_path))
            print(file_name)
            PET_1_path = os.path.join(file_base_data,file_name)
            if not os.path.exists(PET_1_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join(file_base,'Normal',file_name)
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists normal!!!!')
                print(gt_path)
                continue
            coe_pet1 = data_reshape(PET_1_path)
            coe_gt = data_reshape(gt_path)
        
            #coe_pet1 = trans_wave(PET1)
            #coe_gt = trans_wave(gt)
            if coe_gt.shape[2] != coe_pet1.shape[2]: 
                print('False!!!shape different!!!!')
                continue
            x,y,j = 0,0,0
            while j < coe_gt.shape[2]:
                x,y = 0,0
                print(j)
                nextstep_j = shape3//4
                if j+shape3+1 < coe_gt.shape[2]:
                    if np.mean(coe_gt[:,:,j:j+shape3])<30/max_img:
                        nextstep_j = shape3//4*3
                    j_start = j
                    j_end = j+shape3
                else:
                    j_start = coe_gt.shape[2]-shape3
                    j_end = coe_gt.shape[2]
                    j = coe_gt.shape[2]
                    '''PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break'''
                while y < coe_gt.shape[1]:
                    x = 0
                    print(y)
                    nextstep_y = shape3//4
                    if y+shape3+1 < coe_gt.shape[1]:
                        if np.mean(coe_gt[:,y:y+shape3,j_start:j_end])<30/max_img:
                            nextstep_y = shape3//4*3
                        y_start = y
                        y_end = y+shape3
                    else:
                        y_start = coe_gt.shape[1]-shape3
                        y_end = coe_gt.shape[1]
                        y = coe_gt.shape[1]
                    while x < coe_gt.shape[0]:
                        print(x)
                        nextstep_x = shape3//4
                        if x+shape3+1 < coe_gt.shape[0]:
                            if np.mean(coe_gt[x:x+shape3,y_start:y_end,j_start:j_end])<30/max_img:
                                nextstep_x = shape3//4*3
                            x_start = x
                            x_end = x+shape3
                        else:
                            x_start = coe_gt.shape[0]-shape3
                            x_end = coe_gt.shape[0]
                            x = coe_gt.shape[0]
                        PET_pool_1.append(np.expand_dims(coe_pet1[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        Ground_truth_pool.append(np.expand_dims(coe_gt[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        out_filename = "filenames" + '+{}+{}+{}+{}'.format(file_name, x,y,j)
                        filenames_pool.append(np.expand_dims([out_filename], axis=0))
                        x = x+nextstep_x
                    y = y_end+nextstep_y
                j = j_end+nextstep_j
                
def write_64_3d_test(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 64
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,shape3,shape3,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,shape3,shape3,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)


    for niifile in normal_path_list:
        if 1:
            file_name = niifile
            cdata = pd.read_csv("/media/data/fanxuan/meta_info.csv")
            drf = np.array(cdata['DRF'][ cdata['PID'].isin([file_name]) ])
            file_base = '/media/data/fanxuan/data/'
            print(file_name)
            PET_1_path = os.path.join(file_base,'test',file_name)
            for niifile in os.listdir(PET_1_path):
                PET_1_path = os.path.join(PET_1_path,niifile)
            if 'nii.gz' not in PET_1_path:
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join(file_base,'normal',file_name)
            for niifile in os.listdir(gt_path):
                gt_path = os.path.join(gt_path,niifile)
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists normal!!!!')
                print(gt_path)
                continue

            coe_pet1 = data_reshape(PET_1_path)
            coe_gt = data_reshape(gt_path)
        
            #coe_pet1 = trans_wave(PET1)
            #coe_gt = trans_wave(gt)
            if coe_gt.shape[2] != coe_pet1.shape[2]: 
                print('False!!!shape different!!!!')
                continue
            x,y,j = 0,0,0
            while j < coe_gt.shape[2]:
                x,y = 0,0
                print(j)
                nextstep_j = shape3//4*3
                if j+shape3+1 < coe_gt.shape[2]:
                    if np.mean(coe_gt[:,:,j:j+shape3])<30/max_img:
                        nextstep_j = shape3//4*3
                    j_start = j
                    j_end = j+shape3
                else:
                    j_start = coe_gt.shape[2]-shape3
                    j_end = coe_gt.shape[2]
                    j = coe_gt.shape[2]
                    '''PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break'''
                while y < coe_gt.shape[1]:
                    x = 0
                    print(y)
                    nextstep_y = shape3//4*3
                    if y+shape3+1 < coe_gt.shape[1]:
                        if np.mean(coe_gt[:,y:y+shape3,j_start:j_end])<30/max_img:
                            nextstep_y = shape3//4*3
                        y_start = y
                        y_end = y+shape3
                    else:
                        y_start = coe_gt.shape[1]-shape3
                        y_end = coe_gt.shape[1]
                        y = coe_gt.shape[1]
                    while x < coe_gt.shape[0]:
                        print(x)
                        nextstep_x = shape3//4*3
                        if x+shape3+1 < coe_gt.shape[0]:
                            if np.mean(coe_gt[x:x+shape3,y_start:y_end,j_start:j_end])<30/max_img:
                                nextstep_x = shape3//4*3
                            x_start = x
                            x_end = x+shape3
                        else:
                            x_start = coe_gt.shape[0]-shape3
                            x_end = coe_gt.shape[0]
                            x = coe_gt.shape[0]
                        PET_pool_1.append(np.expand_dims(coe_pet1[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        Ground_truth_pool.append(np.expand_dims(coe_gt[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        out_filename = "filenames" + '+{}+{}+{}+{}'.format(file_name, x,y,j)
                        filenames_pool.append(np.expand_dims([out_filename], axis=0))
                        x = x+nextstep_x
                    y = y+nextstep_y
                j = j+nextstep_j
                
def write_h5_3d_64(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 64
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_2_dose',tables.Float32Atom(),
                                          shape=(0,shape3,shape3,shape3),expectedrows=10000000)
    PET_pool_2 = target_file.create_earray(target_file.root,dataset+'_4_dose',tables.Float32Atom(),
                                          shape=(0,shape3,shape3,shape3),expectedrows=10000000)
    PET_pool_3 = target_file.create_earray(target_file.root,dataset+'_10_dose',tables.Float32Atom(),
                                          shape=(0,shape3,shape3,shape3),expectedrows=10000000)
    PET_pool_4 = target_file.create_earray(target_file.root,dataset+'_20_dose',tables.Float32Atom(),
                                          shape=(0,shape3,shape3,shape3),expectedrows=10000000)
    PET_pool_5 = target_file.create_earray(target_file.root,dataset+'_50_dose',tables.Float32Atom(),
                                          shape=(0,shape3,shape3,shape3),expectedrows=10000000)
    PET_pool_6 = target_file.create_earray(target_file.root,dataset+'_100_dose',tables.Float32Atom(),
                                          shape=(0,shape3,shape3,shape3),expectedrows=10000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                   shape=(0,shape3,shape3,shape3),expectedrows=10000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=10000000)



    for normal_path in normal_path_list:
        if 1:
            file_name = os.path.basename(normal_path)
            file_base = os.path.dirname(os.path.dirname(normal_path))
            print(file_name)
            PET_1_path = os.path.join(file_base,'2_dose',file_name)
            if not os.path.exists(PET_1_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            PET_2_path = os.path.join(file_base,'4_dose',file_name)
            if not os.path.exists(PET_2_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            PET_3_path = os.path.join(file_base,'10_dose',file_name)
            if not os.path.exists(PET_3_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            PET_4_path = os.path.join(file_base,'20_dose',file_name)
            if not os.path.exists(PET_4_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            PET_5_path = os.path.join(file_base,'50_dose',file_name)
            if not os.path.exists(PET_5_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            PET_6_path = os.path.join(file_base,'100_dose',file_name)
            if not os.path.exists(PET_6_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join(file_base,'Normal',file_name)
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists test!!!!')
                continue
            coe_pet1 = data_reshape(PET_1_path)
            coe_pet2 = data_reshape(PET_2_path)
            coe_pet3 = data_reshape(PET_3_path)
            coe_pet4 = data_reshape(PET_4_path)
            coe_pet5 = data_reshape(PET_5_path)
            coe_pet6 = data_reshape(PET_6_path)
            coe_gt = data_reshape(gt_path)
            if coe_gt.shape[2] != coe_pet1.shape[2] or coe_gt.shape[2] != coe_pet2.shape[2] or coe_gt.shape[2] != coe_pet3.shape[2] or coe_gt.shape[2] != coe_pet4.shape[2] or coe_gt.shape[2] != coe_pet5.shape[2] or coe_gt.shape[2] != coe_pet6.shape[2]: 
                continue            
        
            #coe_pet1 = trans_wave(PET1)
            #coe_gt = trans_wave(gt)
            if coe_gt.shape[2] != coe_pet1.shape[2]: 
                print('False!!!shape different!!!!')
                continue
            x,y,j = 0,0,0
            while j < coe_gt.shape[2]:
                x,y = 0,0
                print(j)
                nextstep_j = shape3//4*3
                if j+shape3+1 < coe_gt.shape[2]:
                    if np.mean(coe_gt[:,:,j:j+shape3])<30/max_img:
                        nextstep_j = shape3//4*3
                    j_start = j
                    j_end = j+shape3
                else:
                    j_start = coe_gt.shape[2]-shape3
                    j_end = coe_gt.shape[2]
                    j = coe_gt.shape[2]
                    '''PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break'''
                while y < coe_gt.shape[1]:
                    x = 0
                    print(y)
                    nextstep_y = shape3//4*3
                    if y+shape3+1 < coe_gt.shape[1]:
                        if np.mean(coe_gt[:,y:y+shape3,j_start:j_end])<30/max_img:
                            nextstep_y = shape3//4*3
                        y_start = y
                        y_end = y+shape3
                    else:
                        y_start = coe_gt.shape[1]-shape3
                        y_end = coe_gt.shape[1]
                        y = coe_gt.shape[1]
                    while x < coe_gt.shape[0]:
                        print(x)
                        nextstep_x = shape3//4*3
                        if x+shape3+1 < coe_gt.shape[0]:
                            if np.mean(coe_gt[x:x+shape3,y_start:y_end,j_start:j_end])<30/max_img:
                                nextstep_x = shape3//4*3
                            x_start = x
                            x_end = x+shape3
                        else:
                            x_start = coe_gt.shape[0]-shape3
                            x_end = coe_gt.shape[0]
                            x = coe_gt.shape[0]
                        PET_pool_1.append(np.expand_dims(coe_pet1[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        PET_pool_2.append(np.expand_dims(coe_pet2[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        PET_pool_3.append(np.expand_dims(coe_pet3[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        PET_pool_4.append(np.expand_dims(coe_pet4[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        PET_pool_5.append(np.expand_dims(coe_pet5[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        PET_pool_6.append(np.expand_dims(coe_pet6[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        Ground_truth_pool.append(np.expand_dims(coe_gt[x_start:x_end,y_start:y_end,j_start:j_end], axis=0))
                        out_filename = "filenames" + '+{}+{}+{}+{}'.format(file_name, x,y,j)
                        filenames_pool.append(np.expand_dims([out_filename], axis=0))
                        x = x+nextstep_x
                    y = y+nextstep_y
                j = j+nextstep_j



'''def write_h5_3d_wave(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    #uexplorer
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_2_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_2 = target_file.create_earray(target_file.root,dataset+'_4_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_3 = target_file.create_earray(target_file.root,dataset+'_10_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_4 = target_file.create_earray(target_file.root,dataset+'_20_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_5 = target_file.create_earray(target_file.root,dataset+'_50_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_6 = target_file.create_earray(target_file.root,dataset+'_100_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=10000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=10000000)



    for normal_path in normal_path_list:
        if 1:
            file_name = os.path.basename(normal_path)
            file_base = os.path.dirname(os.path.dirname(normal_path))
            print(file_name)
            PET_1_path = os.path.join(file_base,'2_dose',file_name)
            if not os.path.exists(PET_1_path):
                continue
            PET_2_path = os.path.join(file_base,'4_dose',file_name)
            if not os.path.exists(PET_2_path):
                continue
            PET_3_path = os.path.join(file_base,'10_dose',file_name)
            if not os.path.exists(PET_3_path):
                continue
            PET_4_path = os.path.join(file_base,'20_dose',file_name)
            if not os.path.exists(PET_4_path):
                continue
            PET_5_path = os.path.join(file_base,'50_dose',file_name)
            if not os.path.exists(PET_5_path):
                continue
            PET_6_path = os.path.join(file_base,'100_dose',file_name)
            if not os.path.exists(PET_6_path):
                continue
            gt_path = os.path.join(file_base,'Normal',file_name)
            if not os.path.exists(gt_path):
                continue
            PET1 = data_reshape(PET_1_path)
            PET2 = data_reshape(PET_2_path)
            PET3 = data_reshape(PET_3_path)
            PET4 = data_reshape(PET_4_path)
            PET5 = data_reshape(PET_5_path)
            PET6 = data_reshape(PET_6_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_pet2 = trans_wave(PET2)
            coe_pet3 = trans_wave(PET3)
            coe_pet4 = trans_wave(PET4)
            coe_pet5 = trans_wave(PET5) 
            coe_pet6 = trans_wave(PET6)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3] or coe_gt.shape[3] != coe_pet2.shape[3] or coe_gt.shape[3] != coe_pet3.shape[3] or coe_gt.shape[3] != coe_pet4.shape[3] or coe_gt.shape[3] != coe_pet5.shape[3] or coe_gt.shape[3] != coe_pet6.shape[3]: 
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,j:j+shape3], axis=0))
                PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,j:j+shape3], axis=0))
                PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,j:j+shape3], axis=0))
                PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,j:j+shape3], axis=0))
                PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep

'''

def write_h5_3d_wave(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    #subject
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_2_dose',tables.Float32Atom(),
                                         shape=(0,8,224,224,shape3),expectedrows=10000000)
    PET_pool_2 = target_file.create_earray(target_file.root,dataset+'_4_dose',tables.Float32Atom(),
                                         shape=(0,8,224,224,shape3),expectedrows=10000000)
    PET_pool_3 = target_file.create_earray(target_file.root,dataset+'_10_dose',tables.Float32Atom(),
                                         shape=(0,8,224,224,shape3),expectedrows=10000000)
    PET_pool_4 = target_file.create_earray(target_file.root,dataset+'_20_dose',tables.Float32Atom(),
                                         shape=(0,8,224,224,shape3),expectedrows=10000000)
    PET_pool_5 = target_file.create_earray(target_file.root,dataset+'_50_dose',tables.Float32Atom(),
                                         shape=(0,8,224,224,shape3),expectedrows=10000000)
    PET_pool_6 = target_file.create_earray(target_file.root,dataset+'_100_dose',tables.Float32Atom(),
                                         shape=(0,8,224,224,shape3),expectedrows=10000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,224,224,shape3),expectedrows=10000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=10000000)



    for normal_path in normal_path_list:
        if 1:
            file_name = os.path.basename(normal_path)
            file_base = os.path.dirname(os.path.dirname(normal_path))
            print(file_name)
            PET_1_path = os.path.join(file_base,'2_dose',file_name)
            if not os.path.exists(PET_1_path):
                continue
            PET_2_path = os.path.join(file_base,'4_dose',file_name)
            if not os.path.exists(PET_2_path):
                continue
            PET_3_path = os.path.join(file_base,'10_dose',file_name)
            if not os.path.exists(PET_3_path):
                continue
            PET_4_path = os.path.join(file_base,'20_dose',file_name)
            if not os.path.exists(PET_4_path):
                continue
            PET_5_path = os.path.join(file_base,'50_dose',file_name)
            if not os.path.exists(PET_5_path):
                continue
            PET_6_path = os.path.join(file_base,'100_dose',file_name)
            if not os.path.exists(PET_6_path):
                continue
            gt_path = os.path.join(file_base,'Normal',file_name)
            if not os.path.exists(gt_path):
                continue
            PET1 = data_reshape(PET_1_path)
            PET2 = data_reshape(PET_2_path)
            PET3 = data_reshape(PET_3_path)
            PET4 = data_reshape(PET_4_path)
            PET5 = data_reshape(PET_5_path)
            PET6 = data_reshape(PET_6_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_pet2 = trans_wave(PET2)
            coe_pet3 = trans_wave(PET3)
            coe_pet4 = trans_wave(PET4)
            coe_pet5 = trans_wave(PET5) 
            coe_pet6 = trans_wave(PET6)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3] or coe_gt.shape[3] != coe_pet2.shape[3] or coe_gt.shape[3] != coe_pet3.shape[3] or coe_gt.shape[3] != coe_pet4.shape[3] or coe_gt.shape[3] != coe_pet5.shape[3] or coe_gt.shape[3] != coe_pet6.shape[3]: 
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,j:j+shape3], axis=0))
                PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,j:j+shape3], axis=0))
                PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,j:j+shape3], axis=0))
                PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,j:j+shape3], axis=0))
                PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep
                
                
                
                
'''def write_test_3d_wave(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)


    for niifile in normal_path_list:
        file_name = niifile
        cdata = pd.read_csv("/media/data/fanxuan/meta_info.csv")
        drf = np.array(cdata['DRF'][ cdata['PID'].isin([file_name]) ])
        if 'Anonymous' not in file_name:

            file_base = '/media/data/fanxuan/data/'
            print(file_name)
            PET_1_path = os.path.join(file_base,'test',file_name)
            for niifile in os.listdir(PET_1_path):
                PET_1_path = os.path.join(PET_1_path,niifile)
            if 'nii.gz' not in PET_1_path:
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join(file_base,'normal',file_name)
            for niifile in os.listdir(gt_path):
                gt_path = os.path.join(gt_path,niifile)
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists normal!!!!')
                print(gt_path)
                continue
            PET1 = data_reshape(PET_1_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3]: 
                print('False!!!shape different!!!!')
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}+{}'.format(file_name,drf, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep 
                
'''        
def write_test_3d_wave(target_data_dir,dataset,target_file,normal_path_list):
    start = 0
    shape3 = 16
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,8,224,224,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,224,224,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)


    for niifile in normal_path_list:
        file_name = niifile
        cdata = pd.read_csv("/media/data/fanxuan/meta_info.csv")
        drf = np.array(cdata['DRF'][ cdata['PID'].isin([file_name]) ])
        #if 'Anonymous' in file_name and drf != 'real_reduced_dose':
        #if drf == 'real_reduced_dose':
        if 'Anonymous' not in file_name:
            file_base = '/media/data/fanxuan/data/'
            print(file_name)
            PET_1_path = os.path.join(file_base,'test',file_name)
            for niifile in os.listdir(PET_1_path):
                PET_1_path = os.path.join(PET_1_path,niifile)
            if 'nii.gz' not in PET_1_path:
                print('False!!!not os.path.exists test!!!!')
                continue
            gt_path = os.path.join(file_base,'normal',file_name)
            for niifile in os.listdir(gt_path):
                gt_path = os.path.join(gt_path,niifile)
            if not os.path.exists(gt_path):
                print('False!!!not os.path.exists normal!!!!')
                print(gt_path)
                continue
            PET1 = data_reshape(PET_1_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3]: 
                print('False!!!shape different!!!!')
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(file_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}+{}'.format(file_name,drf, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep 
                       
def write_test_3d_wave_all(data_base,dataset,target_file,all_path_list):
    start = 0
    shape3 = 16
    PET_pool_1 = target_file.create_earray(target_file.root,dataset+'_2_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_2 = target_file.create_earray(target_file.root,dataset+'_4_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_3 = target_file.create_earray(target_file.root,dataset+'_10_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_4 = target_file.create_earray(target_file.root,dataset+'_20_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_5 = target_file.create_earray(target_file.root,dataset+'_50_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    PET_pool_6 = target_file.create_earray(target_file.root,dataset+'_100_dose',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=10000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=10000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=10000000)

    for data_name in all_path_list:
        if 'Anonymous' in data_name:
            file_name = 'ori.nii.gz'
            file_base = os.path.join(data_base,data_name)
            print(file_base)
            PET_1_path = os.path.join(file_base,'2_dose',file_name)
            if not os.path.exists(PET_1_path):
                print(PET_1_path)
                continue
            PET_2_path = os.path.join(file_base,'4_dose',file_name)
            if not os.path.exists(PET_2_path):
                continue
            PET_3_path = os.path.join(file_base,'10_dose',file_name)
            if not os.path.exists(PET_3_path):
                continue
            PET_4_path = os.path.join(file_base,'20_dose',file_name)
            if not os.path.exists(PET_4_path):
                continue
            PET_5_path = os.path.join(file_base,'50_dose',file_name)
            if not os.path.exists(PET_5_path):
                continue
            PET_6_path = os.path.join(file_base,'100_dose',file_name)
            if not os.path.exists(PET_6_path):
                continue
            gt_path = os.path.join(file_base,'Normal',file_name)
            if not os.path.exists(gt_path):
                continue
            PET1 = data_reshape(PET_1_path)
            PET2 = data_reshape(PET_2_path)
            PET3 = data_reshape(PET_3_path)
            PET4 = data_reshape(PET_4_path)
            PET5 = data_reshape(PET_5_path)
            PET6 = data_reshape(PET_6_path)
            gt = data_reshape(gt_path)
        
            coe_pet1 = trans_wave(PET1)
            coe_pet2 = trans_wave(PET2)
            coe_pet3 = trans_wave(PET3)
            coe_pet4 = trans_wave(PET4)
            coe_pet5 = trans_wave(PET5) 
            coe_pet6 = trans_wave(PET6)
            coe_gt = trans_wave(gt)
            if coe_gt.shape[3] != coe_pet1.shape[3] or coe_gt.shape[3] != coe_pet2.shape[3] or coe_gt.shape[3] != coe_pet3.shape[3] or coe_gt.shape[3] != coe_pet4.shape[3] or coe_gt.shape[3] != coe_pet5.shape[3] or coe_gt.shape[3] != coe_pet6.shape[3]: 
                print('not fix')
                continue
            j = 0
            while j < coe_gt.shape[3]:
                print(j)
                nextstep = shape3//4
                if j+shape3+1 < coe_gt.shape[3]:
                    if np.mean(coe_gt[:,:,:,j:j+shape3])<30/max_img:
                        nextstep = shape3//4*3
                else:
                    PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                    out_filename = "filenames" + '+{}+{}'.format(data_name, j)
                    filenames_pool.append(np.expand_dims([out_filename], axis=0))
                    break
                PET_pool_1.append(np.expand_dims(coe_pet1[:,:,:,j:j+shape3], axis=0))
                PET_pool_2.append(np.expand_dims(coe_pet2[:,:,:,j:j+shape3], axis=0))
                PET_pool_3.append(np.expand_dims(coe_pet3[:,:,:,j:j+shape3], axis=0))
                PET_pool_4.append(np.expand_dims(coe_pet4[:,:,:,j:j+shape3], axis=0))
                PET_pool_5.append(np.expand_dims(coe_pet5[:,:,:,j:j+shape3], axis=0))
                PET_pool_6.append(np.expand_dims(coe_pet6[:,:,:,j:j+shape3], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j:j+shape3], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(data_name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                j = j+nextstep


           


path = '/media/data/fanxuan/data/data_all_nii/'
######################test
path_test = '/media/data/fanxuan/data/Explorer_test_nii/'
test_name_list = find_filepath_test()
print(test_name_list)

save_path = os.path.join('/media/data/fanxuan/data','data'+'_{}'.format("h5data"))
if not os.path.exists(save_path):
    os.makedirs(save_path)
target_file = tables.open_file(os.path.join(save_path, 'data_wave_test_subject_224.h5'), mode='w')
#df = pd.read_csv("/home/xuan/code/drf100.csv")
#test_name_list = df['Name']

#write_test_3d(path, 'test', target_file, test_list)
#write_test_3d_d(path, 'test', target_file, test_name_list)
write_test_3d_wave(path, 'test', target_file, test_name_list)
#write_test_3d_wave_all(path_test, 'test', target_file, test_name_list)
target_file.close()
'''
######################normal
#normal_path_list = find_filepath_old('100_dose')
normal_path_list = find_filepath()
random.shuffle(normal_path_list)

print(normal_path_list)
if 1:
    train_list = normal_path_list[:int(0.8 * len(normal_path_list))]
    valid_list = normal_path_list[int(0.8 * len(normal_path_list)):]
    save_path = os.path.join('/media/data/fanxuan/data','data'+'_{}'.format("h5data"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    target_file = tables.open_file(os.path.join(save_path, 'data_wave_Subject.h5'), mode='w')
    write_h5_3d_wave(path, 'train', target_file, train_list)
    write_h5_3d_wave(path, 'valid', target_file, valid_list)
    target_file.close() 
'''