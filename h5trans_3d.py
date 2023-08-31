import os
import numpy as np
import dicom2nifti
import shutil
import h5py
import tables
import nibabel as nib
from sklearn.model_selection import KFold
import logging
import time
import SimpleITK as sitk
import pydicom


import pywt
import pywt.data

max_img=612363.6
#max_img=1172342.5
max_all_img = 10312549
 
def write_h5_normal_3d(target_data_dir,dataset,target_file,index):
    start = 0
    shape3 = 16
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,360,360,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,360,360,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        for j in range(PET.shape[2]//(shape3//4)):
            if j*(shape3//4)+shape3 >= PET.shape[2]:
                PET_pool.append(np.expand_dims(PET[:,:,PET.shape[2]-shape3:PET.shape[2]], axis=0))
                Ground_truth_pool.append(np.expand_dims(gt[:,:,gt.shape[2]-shape3:gt.shape[2]], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                break
            PET_pool.append(np.expand_dims(PET[:,:,j*(shape3//4):j*(shape3//4)+shape3], axis=0))
            Ground_truth_pool.append(np.expand_dims(gt[:,:,j*(shape3//4):j*(shape3//4)+shape3], axis=0))
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))

def write_h5_3d(target_data_dir,dataset,target_file,index):
    start = 0
    shape3 = 16
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        PET = PET[(PET.shape[0]//2-169):(PET.shape[0]//2+169), (PET.shape[1]//2-169):(PET.shape[1]//2+169)]
        coe = pywt.dwtn(PET, 'haar', mode='symmetric', axes=None)
        
        coe_pet = np.array([coe['aaa'],coe['aad'],coe['ada'],coe['add'],coe['daa'],coe['dad'],coe['dda'],coe['ddd']])
        print(coe_pet.shape)
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-169):(gt.shape[0]//2+169), (gt.shape[1]//2-169):(gt.shape[1]//2+169)]
        coe = pywt.dwtn(gt, 'haar', mode='symmetric', axes=None)
        coe_gt = np.array([coe['aaa'],coe['aad'],coe['ada'],coe['add'],coe['daa'],coe['dad'],coe['dda'],coe['ddd']])
        for j in range(coe_gt.shape[3]//(shape3//4)):
            if j*(shape3//4)+shape3 >= coe_gt.shape[3]:
                PET_pool.append(np.expand_dims(coe_pet[:,:,:,coe_pet.shape[3]-shape3:coe_pet.shape[3]], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                break
            PET_pool.append(np.expand_dims(coe_pet[:,:,:,j*(shape3//4):j*(shape3//4)+shape3], axis=0))
            Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j*(shape3//4):j*(shape3//4)+shape3], axis=0))
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))

def write_h5_all(target_data_dir,dataset,target_file,index):
    start = 0
    shape3 = 16
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,8,176,176,shape3),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,8,176,176,shape3),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        PET = PET[(PET.shape[0]//2-169):(PET.shape[0]//2+169), (PET.shape[1]//2-169):(PET.shape[1]//2+169)]
        coe = pywt.dwtn(PET, 'haar', mode='symmetric', axes=None)
        
        coe_pet = np.array([coe['aaa'],coe['aad'],coe['ada'],coe['add'],coe['daa'],coe['dad'],coe['dda'],coe['ddd']])
        print(coe_pet.shape)
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-169):(gt.shape[0]//2+169), (gt.shape[1]//2-169):(gt.shape[1]//2+169)]
        coe = pywt.dwtn(gt, 'haar', mode='symmetric', axes=None)
        coe_gt = np.array([coe['aaa'],coe['aad'],coe['ada'],coe['add'],coe['daa'],coe['dad'],coe['dda'],coe['ddd']])
        for j in range(coe_gt.shape[3]//(shape3//4)):
            if j*(shape3//4)+shape3 >= coe_gt.shape[3]:
                PET_pool.append(np.expand_dims(coe_pet[:,:,:,coe_pet.shape[3]-shape3:coe_pet.shape[3]], axis=0))
                Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,coe_gt.shape[3]-shape3:coe_gt.shape[3]], axis=0))
                out_filename = "filenames" + '+{}+{}'.format(name, j)
                filenames_pool.append(np.expand_dims([out_filename], axis=0))
                break
            PET_pool.append(np.expand_dims(coe_pet[:,:,:,j*(shape3//4):j*(shape3//4)+shape3], axis=0))
            Ground_truth_pool.append(np.expand_dims(coe_gt[:,:,:,j*(shape3//4):j*(shape3//4)+shape3], axis=0))
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))


path = "/media/data/fanxuan/data/PART2"
target_path = "/media/data/fanxuan/data"
'''

path = "/media/data/fanxuan/data/PART2_Siemens"
target_path = "/media/data/fanxuan/data"
'''
n = 0
wavelet = 'haar'
'''
path = '/media/data/fanxuan/data/uExplorerPART2'
target_path = '/media/data/fanxuan/data/PART2'
for root, dirs, files in os.walk(path):
    if files == [] and 'Anonymous' in root:
        n = n+1
    if dirs == []:
        print('root:{} n:{}'.format(root, n))
        file_to_nii(root, n, target_path)
'''
'''
index = np.arange(1,len([x for x in os.listdir(path) if not x.startswith('.')])+1)
cross_validation = False
if cross_validation == False:
    train_index = index[:int(0.8 * len(index))]
    valid_index = index[int(0.8 * len(index)):]
    save_path = os.path.join(os.path.dirname(path), os.path.basename(path)+'_{}'.format("h5data"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    target_file_LL = tables.open_file(os.path.join(save_path, 'data_LL.h5'), mode='w')
    target_file_LH = tables.open_file(os.path.join(save_path, 'data_LH.h5'), mode='w')
    target_file_HL = tables.open_file(os.path.join(save_path, 'data_HL.h5'), mode='w')
    target_file_HH = tables.open_file(os.path.join(save_path, 'data_HH.h5'), mode='w')
    target_data_dir = target_path

    write_h5_wave(target_data_dir, 'train', target_file_LL, target_file_LH, target_file_HL, target_file_HH, train_index, wavelet)
    write_h5_wave(target_data_dir, 'valid', target_file_LL, target_file_LH, target_file_HL, target_file_HH, valid_index, wavelet)
    target_file_LL.close()
    target_file_LH.close()
    target_file_HL.close()
    target_file_HH.close()
'''
'''
index = np.arange(1,len([x for x in os.listdir("/media/data/fanxuan/data/PART2_Siemens/Normal") if not x.startswith('.')])+1)
cross_validation = False
if cross_validation == False:
    train_index = index[:int(0.8 * len(index))]
    valid_index = index[int(0.8 * len(index)):]
    save_path = os.path.join('/media/data/fanxuan/data', os.path.basename(path)+'_{}'.format("h5data"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    target_file = tables.open_file(os.path.join(save_path, 'data_Siemens_wave_50_3d.h5'), mode='w')
    target_data_dir = '/media/data/fanxuan/data/PART2_Siemens'
    #write_h5_3d(target_data_dir, 'train', target_file, train_index)
    write_h5_3d(target_data_dir, 'valid', target_file, index)

    target_file.close()
    '''
index = np.arange(1,len([x for x in os.listdir("/media/data/fanxuan/data/PART2/Normal") if not x.startswith('.')])+1)
cross_validation = False
if cross_validation == False:
    train_index = index[:int(0.8 * len(index))]
    valid_index = index[int(0.8 * len(index)):]
    save_path = os.path.join('/media/data/fanxuan/data', os.path.basename(path)+'_{}'.format("h5data"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    target_file = tables.open_file(os.path.join(save_path, 'data_wave_10_3d_16.h5'), mode='w')
    target_data_dir = '/media/data/fanxuan/data/PART2'
    write_h5_3d(target_data_dir, 'train', target_file, train_index)
    write_h5_3d(target_data_dir, 'valid', target_file, valid_index)

    target_file.close()
