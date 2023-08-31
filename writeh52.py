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
img_max = 498879.6
max_img = 612363.6000000001 
min_img = 0.0 
max_ll = 1114619.1216900002 
min_ll = 0.0 
max_lh = 192557.671652 
min_lh = -225969.58185200003
max_hl = 174515.240144 
min_hl = -176200.5453 
max_hh = 47016.231371999995 
min_hh = -44065.89666000002
img_range = max_img - min_img
ll_range = max_ll - min_ll
lh_range = max_lh - min_lh
hl_range = max_hl - min_hl
hh_range = max_hh - min_hh
 
def file_to_nii(path_basic, index, target_path):
    def dcm2nii(dcms_path, nii_path):
 
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
        reader.SetFileNames(dicom_names)
        image2 = reader.Execute()

        image_array = sitk.GetArrayFromImage(image2)  
        origin = image2.GetOrigin()  
        spacing = image2.GetSpacing()  
        direction = image2.GetDirection()  
     
        image3 = sitk.GetImageFromArray(image_array)
        image3.SetSpacing(spacing)
        image3.SetDirection(direction)
        image3.SetOrigin(origin)
        sitk.WriteImage(image3, nii_path)
    fold_name = os.path.basename(path_basic)
    if "normal" in fold_name:
        target_path = os.path.join(target_path, "Normal" )
    if "1-10 dose" in fold_name:
        target_path = os.path.join(target_path, "10_dose" )
    if "1-20 dose" in fold_name:
        target_path = os.path.join(target_path, "20_dose" )
    if "1-50 dose" in fold_name:
        target_path = os.path.join(target_path, "50_dose" )
    if "1-100 dose" in fold_name:
        target_path = os.path.join(target_path, "100_dose" )
    if "1-2 dose" in fold_name:
        target_path = os.path.join(target_path, "2_dose" )
    if "1-4 dose" in fold_name:
        target_path = os.path.join(target_path, "4_dose" )
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    nii_path =  os.path.join(target_path, "patient_"+ str(index)+".nii" )
    print(path_basic)
    print(nii_path)
    dcm2nii(path_basic, nii_path)
    
def normalization(img):
    img_range = np.max(img) - np.min(img)
    print(img_range)
    if img_range == 0:
        return img
    else:
        return (img / img_range -1)*2


def write_h5(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,360,360),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,360,360),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        for j in range(PET.shape[2]):
            PET_pool.append(np.expand_dims(PET[:,:,j], axis=0))
            Ground_truth_pool.append(np.expand_dims(gt[:,:,j], axis=0))
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
def write_h5_two(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_dose',tables.Float32Atom(),
                                         shape=(0,360,360),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,360,360),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_1_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET_1 = np.array(nib.load(PET_1_path).dataobj)
        PET_1 = PET_1/max_img
        PET_2_path = os.path.join(target_data_dir,'100_dose',"patient_"+ str(name)+".nii")
        PET_2 = np.array(nib.load(PET_2_path).dataobj)
        PET_2 = PET_2/max_img
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        for j in range(PET_1.shape[2]):
            PET_pool.append(np.expand_dims(PET_1[:,:,j], axis=0))
            PET_pool.append(np.expand_dims(PET_2[:,:,j], axis=0))
            Ground_truth_pool.append(np.expand_dims(gt[:,:,j], axis=0))
            Ground_truth_pool.append(np.expand_dims(gt[:,:,j], axis=0))
            out_filename = "filenames_fir" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
            out_filename = "filenames_sec" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))

def write_h5_wave(target_data_dir,dataset,target_file_LL, target_file_LH, target_file_HL, target_file_HH, index, wavelet):
    start = 0
    PET_pool_LL = target_file_LL.create_earray(target_file_LL.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_LL = target_file_LL.create_earray(target_file_LL.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_LL = target_file_LL.create_earray(target_file_LL.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    
    PET_pool_LH = target_file_LH.create_earray(target_file_LH.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_LH = target_file_LH.create_earray(target_file_LH.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_LH = target_file_LH.create_earray(target_file_LH.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    PET_pool_HL = target_file_HL.create_earray(target_file_HL.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_HL = target_file_HL.create_earray(target_file_HL.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_HL = target_file_HL.create_earray(target_file_HL.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    PET_pool_HH = target_file_HH.create_earray(target_file_HH.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_HH = target_file_HH.create_earray(target_file_HH.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_HH = target_file_HH.create_earray(target_file_HH.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]

        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            LL_pet = LL_pet/ll_range
            LH_pet = (LH_pet/lh_range)*2
            HL_pet = (HL_pet/hl_range)*2
            HH_pet = (HH_pet/hh_range)*2
            PET_pool_LL.append(np.expand_dims(LL_pet, axis=0))
            PET_pool_LH.append(np.expand_dims(LH_pet, axis=0))
            PET_pool_HL.append(np.expand_dims(HL_pet, axis=0))
            PET_pool_HH.append(np.expand_dims(HH_pet, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            LL_gt = LL_gt/ll_range
            LH_gt = (LH_gt/lh_range)*2
            HL_gt = (HL_gt/hl_range)*2
            HH_gt = (HH_gt/hh_range)*2
            Ground_truth_pool_LL.append(np.expand_dims(LL_gt, axis=0))
            Ground_truth_pool_LH.append(np.expand_dims(LH_gt, axis=0))
            Ground_truth_pool_HL.append(np.expand_dims(HL_gt, axis=0))
            Ground_truth_pool_HH.append(np.expand_dims(HH_gt, axis=0))
            
            
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool_LL.append(np.expand_dims([out_filename], axis=0))
            filenames_pool_LH.append(np.expand_dims([out_filename], axis=0))
            filenames_pool_HL.append(np.expand_dims([out_filename], axis=0))
            filenames_pool_HH.append(np.expand_dims([out_filename], axis=0))
    
def write_h5_4turnel(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,4,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
        
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            LL_pet = LL_pet/ll_range
            LH_pet = LH_pet/lh_range*2
            HL_pet = HL_pet/hl_range*2
            HH_pet = HH_pet/hh_range*2
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            LL_gt = LL_gt/ll_range
            LH_gt = LH_gt/lh_range*2
            HL_gt = HL_gt/hl_range*2
            HH_gt = HH_gt/hh_range*2
            gt_wave = np.array([LL_gt, LH_gt, HL_gt, HH_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))    
def write_h5_3turnel(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,3,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,3,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
        
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            LH_pet = (LH_pet/lh_range)*2
            HL_pet = (HL_pet/hl_range)*2
            HH_pet = (HH_pet/hh_range)*2
            PET_wave = np.array([LH_pet, HL_pet, HH_pet])
            PET_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            LH_gt = (LH_gt/lh_range)*2
            HL_gt = (HL_gt/hl_range)*2
            HH_gt = (HH_gt/hh_range)*2
            gt_wave = np.array([LH_gt, HL_gt, HH_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
            
def write_h5_3wave2(target_data_dir,dataset,target_file,index,rangesign):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        if rangesign == 0:
          PET = PET/img_max
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
        
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        if rangesign == 0:
          gt = gt/img_max
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            print('j:{}'.format(j))
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            if rangesign == -1:
                LH_pet = (LH_pet/lh_range)*2
                HL_pet = (HL_pet/hl_range)*2
                HH_pet = (HH_pet/hh_range)*2
            elif rangesign == 1:
                LH_pet = (LH_pet-min_lh)/lh_range
                HL_pet = (HL_pet-min_hl)/hl_range
                HH_pet = (HH_pet-min_hh)/hh_range
            PET_pool.append(np.expand_dims(LH_pet, axis=0))
            PET_pool.append(np.expand_dims(HL_pet, axis=0))
            PET_pool.append(np.expand_dims(HH_pet, axis=0))
            print('PET_pool:{}'.format(PET_pool.shape))
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            if rangesign == -1:
                LH_gt = (LH_gt/lh_range)*2
                HL_gt = (HL_gt/hl_range)*2
                HH_gt = (HH_gt/hh_range)*2
            else:
                LH_gt = (LH_gt-min_lh)/lh_range
                HL_gt = (HL_gt-min_hl)/hl_range
                HH_gt = (HH_gt-min_hh)/hh_range
            Ground_truth_pool.append(np.expand_dims(LH_gt, axis=0))
            Ground_truth_pool.append(np.expand_dims(HL_gt, axis=0))
            Ground_truth_pool.append(np.expand_dims(HH_gt, axis=0))
                
            out_filename = "filenames_lh" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
            out_filename = "filenames_hl" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
            out_filename = "filenames_hh" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))

def write_h5_LL2(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
        
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            print('j:{}'.format(j))
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            LL_pet = LL_pet/ll_range
            PET_pool.append(np.expand_dims(LL_pet, axis=0))

            print('PET_pool:{}'.format(PET_pool.shape))
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            LL_gt = LL_gt/ll_range
            Ground_truth_pool.append(np.expand_dims(LL_gt, axis=0))

                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))

path = "/media/data/fanxuan/data/PART2"
target_path = "/media/data/fanxuan/data"
n = 0
wavelet = 'haar'
'''
for root, dirs, files in os.walk(path):
    if files == [] and 'Anonymous' in root:

        n = n+1
    if dirs == []:
        if  "1-100 dose" in root:
            file_to_nii(root, n, target_path)
'''
'''
index = np.arange(1,len([x for x in os.listdir("/media/data/fanxuan/data/PART2/Normal") if not x.startswith('.')])+1)
cross_validation = False
if cross_validation == False:
    train_index = index[:int(0.8 * len(index))]
    valid_index = index[int(0.8 * len(index)):]
    save_path = os.path.join('/media/data/fanxuan/data', os.path.basename(path)+'_{}'.format("h5data"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    target_file_LL = tables.open_file(os.path.join(save_path, 'data_LL.h5'), mode='w')
    target_file_LH = tables.open_file(os.path.join(save_path, 'data_LH.h5'), mode='w')
    target_file_HL = tables.open_file(os.path.join(save_path, 'data_HL.h5'), mode='w')
    target_file_HH = tables.open_file(os.path.join(save_path, 'data_HH.h5'), mode='w')
    target_data_dir = '/media/data/fanxuan/data/PART2'

    write_h5_wave(target_data_dir, 'train', target_file_LL, target_file_LH, target_file_HL, target_file_HH, train_index, wavelet)
    write_h5_wave(target_data_dir, 'valid', target_file_LL, target_file_LH, target_file_HL, target_file_HH, valid_index, wavelet)
    target_file_LL.close()
    target_file_LH.close()
    target_file_HL.close()
    target_file_HH.close()

'''
index = np.arange(1,len([x for x in os.listdir("/media/data/fanxuan/data/PART2/Normal") if not x.startswith('.')])+1)
cross_validation = False
if cross_validation == False:
    train_index = index[:int(0.8 * len(index))]
    valid_index = index[int(0.8 * len(index)):]
    save_path = os.path.join('/media/data/fanxuan/data', os.path.basename(path)+'_{}'.format("h5data"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    target_file = tables.open_file(os.path.join(save_path, 'data_3w_orignal.h5'), mode='w')
    target_data_dir = '/media/data/fanxuan/data/PART2'
    write_h5_3wave2(target_data_dir, 'train', target_file, train_index,0)
    write_h5_3wave2(target_data_dir, 'valid', target_file, valid_index,0)

    target_file.close()
