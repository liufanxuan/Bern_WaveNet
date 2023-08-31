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

wavelet = 'haar'

def find_minmax_wave(target_data_dir,index):
    max_img, max_ll, max_lh, max_hl, max_hh = [],[],[],[],[]
    min_img, min_ll, min_lh, min_hl, min_hh = [],[],[],[],[]
    for name in index:
        PET_path = os.path.join(target_data_dir,'100_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        for j in range(PET.shape[2]):
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            max_ll.append(np.max(LL_pet))
            max_lh.append(np.max(LH_pet))
            max_hl.append(np.max(HL_pet))
            max_hh.append(np.max(HH_pet))
            min_ll.append(np.min(LL_pet))
            min_lh.append(np.min(LH_pet))
            min_hl.append(np.min(HL_pet))
            min_hh.append(np.min(HH_pet))
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        PET_path2 = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET2 = np.array(nib.load(PET_path2).dataobj)
        for j in range(PET.shape[2]):
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            max_ll.append(np.max(LL_pet))
            max_lh.append(np.max(LH_pet))
            max_hl.append(np.max(HL_pet))
            max_hh.append(np.max(HH_pet))
            min_ll.append(np.min(LL_pet))
            min_lh.append(np.min(LH_pet))
            min_hl.append(np.min(HL_pet))
            min_hh.append(np.min(HH_pet))
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        for j in range(gt.shape[2]):
            coeffs_pet = pywt.dwt2(gt[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            max_ll.append(np.max(LL_pet))
            max_lh.append(np.max(LH_pet))
            max_hl.append(np.max(HL_pet))
            max_hh.append(np.max(HH_pet))
            min_ll.append(np.min(LL_pet))
            min_lh.append(np.min(LH_pet))
            min_hl.append(np.min(HL_pet))
            min_hh.append(np.min(HH_pet))
        max_img.append(np.max(gt))
        min_img.append(np.min(gt))
        print(max_img)
        print(min_img)

    return np.max(max_img),np.min(min_img),np.max(max_ll),np.min(min_ll),np.max(max_lh),np.min(min_lh),np.max(max_hl),np.min(min_hl),np.max(max_hh),np.min(min_hh)
    
    
def find_minmax(path):
    max_img, max_ll, max_lh, max_hl, max_hh = [],[],[],[],[]
    min_img, min_ll, min_lh, min_hl, min_hh = [],[],[],[],[]
    names = os.listdir(path)
    for name in names:
        PET_root = os.path.join(path,name)
        pet_names = os.listdir(PET_root)
        for pet_name in pet_names: 
            pet_file_root = os.path.join(PET_root,pet_name)
            PET = np.array(nib.load(pet_file_root).dataobj)
            c1 = PET[0:(PET.shape[0]//2-218), :]

            c2 = PET[(PET.shape[0]//2+218):, :]
            c3 = PET[:, 0:(PET.shape[0]//2-218)]  
            c4 = PET[:, (PET.shape[0]//2+218):] 
            m = np.max([np.max(c1), np.max(c2), np.max(c3), np.max(c4)])
            
            max_img.append(m)
            if m >= 20:
                print(pet_name)
                print(np.shape(c1))
                print('max image:{} '.format([np.max(c1), np.max(c2), np.max(c3), np.max(c4)]))
    return np.max(max_img)
            #print(pet_file_root)
'''   
    for name in index:
        PET_path = os.path.join(target_data_dir,'100_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        PET_path2 = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path2).dataobj)
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        PET_path2 = os.path.join(target_data_dir,'4_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path2).dataobj)
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        PET_path2 = os.path.join(target_data_dir,'50_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path2).dataobj)
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        PET_path2 = os.path.join(target_data_dir,'20_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path2).dataobj)
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        PET_path2 = os.path.join(target_data_dir,'2_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path2).dataobj)
        max_img.append(np.max(PET))
        min_img.append(np.min(PET))
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        max_img.append(np.max(gt))
        min_img.append(np.min(gt))
        print(max_img)
        print(min_img)

    return np.max(max_img),np.min(min_img)'''

path = '/media/data/fanxuan/data/data_all_nii/Subject/'
for root, dirs, files in os.walk(path):
    if 'Normal' in dirs:
        #print(root)
        index = np.arange(1,len([x for x in os.listdir(os.path.join(root,'Normal')) if not x.startswith('.')])+1)
        #print(index)
        max_img = find_minmax(root)
        print('max image:{} '.format(max_img))
        max_all.append(max_img)
print('max image:{}'.format(np.max(max_all)))
'''
# target_path = '/media/data/fanxuan/data/data_all_nii/Subject/PART7'
max_all, min_all = [],[]
for root, dirs, files in os.walk(path):
    if 'Normal' in dirs:
        print(root)
        index = np.arange(1,len([x for x in os.listdir(os.path.join(root,'Normal')) if not x.startswith('.')])+1)
        print(index)
        max_img, min_img = find_minmax(root)
        print('max image:{} min image:{}'.format(max_img, min_img))
        max_all.append(max_img)
        min_all.append(min_img)
print('max image:{} min image:{}'.format(np.max(max_all), np.min(min_all)))
'''
'''
max_img, min_img,max_ll, min_ll,max_lh, min_lh,max_hl, min_hl,max_hh, min_hh = find_minmax_wave(target_path,index)
print('max image:{} min image:{} max_ll:{} min_ll:{} max_lh:{} min_lh:{} max_hl:{} min_hl:{} max_hh:{} min_hh:{}'.format(max_img, min_img,max_ll, min_ll,max_lh, min_lh,max_hl, min_hl,max_hh, min_hh))

max_img, min_img = find_minmax(target_path,index)
print('max image:{} min image:{}'.format(max_img, min_img))'''