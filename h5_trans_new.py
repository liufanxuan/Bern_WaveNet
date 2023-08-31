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
    PET_pool_LL = target_file_LL.create_earray(target_file_LL.root,dataset+'_dose',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_LL = target_file_LL.create_earray(target_file_LL.root,dataset+'_DRF_10',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_LL = target_file_LL.create_earray(target_file_LL.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    
    PET_pool_LH = target_file_LH.create_earray(target_file_LH.root,dataset+'_dose',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_LH = target_file_LH.create_earray(target_file_LH.root,dataset+'_DRF_10',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_LH = target_file_LH.create_earray(target_file_LH.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    PET_pool_HL = target_file_HL.create_earray(target_file_HL.root,dataset+'_dose',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_HL = target_file_HL.create_earray(target_file_HL.root,dataset+'_DRF_10',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_HL = target_file_HL.create_earray(target_file_HL.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    PET_pool_HH = target_file_HH.create_earray(target_file_HH.root,dataset+'_dose',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool_HH = target_file_HH.create_earray(target_file_HH.root,dataset+'_DRF_10',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool_HH = target_file_HH.create_earray(target_file_HH.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    
    for name in index:
        PET_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]

        gt_path = os.path.join(target_data_dir,'Reduce',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_pool_LL.append(np.expand_dims(LL_pet, axis=0))
            PET_pool_LH.append(np.expand_dims(LH_pet, axis=0))
            PET_pool_HL.append(np.expand_dims(HL_pet, axis=0))
            PET_pool_HH.append(np.expand_dims(HH_pet, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
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
    PET1_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,4,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    mse_100 = []
    for name in index:
    
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt_ori = np.array(nib.load(gt_path).dataobj)
        gt = gt_ori/max_img
        #gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        gt = gt[(gt.shape[0]//2-169):(gt.shape[0]//2+169), (gt.shape[1]//2-169):(gt.shape[1]//2+169)]
        
        PET1_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET1 = np.array(nib.load(PET1_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET1 - gt_ori), axis=-1))
        print('mse100:{}'.format(mse_pet))
        mse_100.append(mse_pet)
        PET1 = PET1/max_img
        #PET1 = PET1[(PET1.shape[0]//2-176):(PET1.shape[0]//2+176), (PET1.shape[1]//2-176):(PET1.shape[1]//2+176)]
        PET1 = PET1[(PET1.shape[0]//2-169):(PET1.shape[0]//2+169), (PET1.shape[1]//2-169):(PET1.shape[1]//2+169)]
        print(PET1.shape)
        print(wavelet)
        
        
        for j in range(PET1.shape[2]):
            
            coeffs_pet = pywt.dwt2(PET1[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            #print(PET_wave.shape)
            PET1_pool.append(np.expand_dims(PET_wave, axis=0))
            
           
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            gt_wave = np.array([LL_gt, LH_gt, HL_gt, HH_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
def write_h5_4turnel_last(target_data_dir,dataset,target_file,index):
    start = 0
    PET1_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176,4),expectedrows=1000000)
    
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176,4),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    mse_100 = []
    for name in index:
    
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt_ori = np.array(nib.load(gt_path).dataobj)
        gt = gt_ori/max_img
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        #gt = gt[(gt.shape[0]//2-169):(gt.shape[0]//2+169), (gt.shape[1]//2-169):(gt.shape[1]//2+169)]
        
        PET1_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET1 = np.array(nib.load(PET1_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET1 - gt_ori), axis=-1))
        print('mse100:{}'.format(mse_pet))
        mse_100.append(mse_pet)
        PET1 = PET1/max_img
        PET1 = PET1[(PET1.shape[0]//2-176):(PET1.shape[0]//2+176), (PET1.shape[1]//2-176):(PET1.shape[1]//2+176)]
        #PET1 = PET1[(PET1.shape[0]//2-169):(PET1.shape[0]//2+169), (PET1.shape[1]//2-169):(PET1.shape[1]//2+169)]
        print(PET1.shape)
        print(wavelet)
        
        
        for j in range(PET1.shape[2]):
            
            coeffs_pet = pywt.dwt2(PET1[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.dstack([LL_pet, LH_pet, HL_pet, HH_pet])
            #print(PET_wave.shape)
            PET1_pool.append(np.expand_dims(PET_wave, axis=0))
            
           
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            gt_wave = np.dstack([LL_gt, LH_gt, HL_gt, HH_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
def write_h5_4turnel_test(target_data_dir,dataset,target_file,index):
    start = 0
    PET1_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,4,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    mse_100 = []
    for name in index:
    
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt_ori = np.array(nib.load(gt_path).dataobj)
        gt = gt_ori/max_img
        #gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        gt = gt[(gt.shape[0]//2-169):(gt.shape[0]//2+169), (gt.shape[1]//2-169):(gt.shape[1]//2+169)]
        
        PET1_path = os.path.join(target_data_dir,'50_dose',"patient_"+ str(name)+".nii")
        PET1 = np.array(nib.load(PET1_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET1 - gt_ori), axis=-1))
        print('mse100:{}'.format(mse_pet))
        mse_100.append(mse_pet)
        PET1 = PET1/max_img
        PET1 = PET1[(PET1.shape[0]//2-169):(PET1.shape[0]//2+169), (PET1.shape[1]//2-169):(PET1.shape[1]//2+169)]
        #PET1 = PET1[(PET1.shape[0]//2-176):(PET1.shape[0]//2+176), (PET1.shape[1]//2-176):(PET1.shape[1]//2+176)]
        
        
        
        for j in range(PET1.shape[2]):
            
            coeffs_pet = pywt.dwt2(PET1[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET1_pool.append(np.expand_dims(PET_wave, axis=0))
            
           
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            gt_wave = np.array([LL_gt, LH_gt, HL_gt, HH_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
    
def write_h5_test(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,360,360),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,360,360),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'50_dose',"patient_"+ str(name)+".nii")
        PET_ori = np.array(nib.load(PET_path).dataobj)
        PET1 = PET_ori/max_img
        PET1 = PET1[(PET1.shape[0]//2-180):(PET1.shape[0]//2+180), (PET1.shape[1]//2-180):(PET1.shape[1]//2+180)]
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET_ori - gt), axis=-1))
        print('mse100:{}'.format(mse_pet))
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-180):(gt.shape[0]//2+180), (gt.shape[1]//2-180):(gt.shape[1]//2+180)]
        for j in range(PET1.shape[2]):
            PET_pool.append(np.expand_dims(PET1[:,:,j], axis=0))
            Ground_truth_pool.append(np.expand_dims(gt[:,:,j], axis=0))
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0)) 
                  
def write_h5_4turnel_multi(target_data_dir,dataset,target_file,index):
    start = 0
    PET1_pool = target_file.create_earray(target_file.root,dataset+'_reduce_100',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    PET2_pool = target_file.create_earray(target_file.root,dataset+'_reduce_50',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    PET3_pool = target_file.create_earray(target_file.root,dataset+'_reduce_20',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    PET4_pool = target_file.create_earray(target_file.root,dataset+'_reduce_10',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    PET5_pool = target_file.create_earray(target_file.root,dataset+'_reduce_4',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    PET6_pool = target_file.create_earray(target_file.root,dataset+'_reduce_2',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,4,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    mse_100,mse_50,mse_20,mse_10,mse_4,mse_2 = [],[],[],[],[],[]
    for name in index:
    
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt_ori = np.array(nib.load(gt_path).dataobj)
        gt = gt_ori/max_img
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        PET1_path = os.path.join(target_data_dir,'100_dose',"patient_"+ str(name)+".nii")
        PET1 = np.array(nib.load(PET1_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET1 - gt_ori), axis=-1))
        print('mse100:{}'.format(mse_pet))
        mse_100.append(mse_pet)
        PET1 = PET1/max_img
        PET1 = PET1[(PET1.shape[0]//2-176):(PET1.shape[0]//2+176), (PET1.shape[1]//2-176):(PET1.shape[1]//2+176)]
        
        PET2_path = os.path.join(target_data_dir,'50_dose',"patient_"+ str(name)+".nii")
        PET2 = np.array(nib.load(PET2_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET2 - gt_ori), axis=-1))
        print('mse50:{}'.format(mse_pet))
        mse_50.append(mse_pet)
        PET2 = PET2/max_img
        PET2 = PET2[(PET2.shape[0]//2-176):(PET2.shape[0]//2+176), (PET2.shape[1]//2-176):(PET2.shape[1]//2+176)]
        
        PET3_path = os.path.join(target_data_dir,'20_dose',"patient_"+ str(name)+".nii")
        PET3 = np.array(nib.load(PET3_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET3 - gt_ori), axis=-1))
        mse_20.append(mse_pet)
        print('mse20:{}'.format(mse_pet))
        PET3 = PET3/max_img
        PET3 = PET3[(PET3.shape[0]//2-176):(PET3.shape[0]//2+176), (PET3.shape[1]//2-176):(PET3.shape[1]//2+176)]
        
        PET4_path = os.path.join(target_data_dir,'10_dose',"patient_"+ str(name)+".nii")
        PET4 = np.array(nib.load(PET4_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET4 - gt_ori), axis=-1))
        mse_10.append(mse_pet)
        print('mse10:{}'.format(mse_pet))
        PET4 = PET4/max_img
        PET4 = PET4[(PET4.shape[0]//2-176):(PET4.shape[0]//2+176), (PET4.shape[1]//2-176):(PET4.shape[1]//2+176)]
        
        PET5_path = os.path.join(target_data_dir,'4_dose',"patient_"+ str(name)+".nii")
        PET5 = np.array(nib.load(PET5_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET5 - gt_ori), axis=-1))
        print('mse4:{}'.format(mse_pet))
        mse_4.append(mse_pet)
        PET5 = PET5/max_img
        PET5 = PET5[(PET5.shape[0]//2-176):(PET5.shape[0]//2+176), (PET5.shape[1]//2-176):(PET5.shape[1]//2+176)]
        
        PET6_path = os.path.join(target_data_dir,'2_dose',"patient_"+ str(name)+".nii")
        PET6 = np.array(nib.load(PET6_path).dataobj)
        mse_pet = np.mean(np.mean(np.square(PET6 - gt_ori), axis=-1))
        print('mse2:{}'.format(mse_pet))
        mse_2.append(mse_pet)
        PET6 = PET6/max_img
        PET6 = PET6[(PET6.shape[0]//2-176):(PET6.shape[0]//2+176), (PET6.shape[1]//2-176):(PET6.shape[1]//2+176)]
        
        
        
        for j in range(PET1.shape[2]):
            
            coeffs_pet = pywt.dwt2(PET1[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET1_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_pet = pywt.dwt2(PET2[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET2_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_pet = pywt.dwt2(PET3[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET3_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_pet = pywt.dwt2(PET4[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_peth
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET4_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_pet = pywt.dwt2(PET5[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET5_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_pet = pywt.dwt2(PET6[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET6_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            gt_wave = np.array([LL_gt, LH_gt, HL_gt, HH_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))
    print(np.mean(mse_100))
    print(np.mean(mse_50))
    print(np.mean(mse_20))
    print(np.mean(mse_10))
    print(np.mean(mse_4))
    print(np.mean(mse_2))

def write_h5_3turnel(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,3,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,3,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'50_dose',"patient_"+ str(name)+".nii")
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
            PET_wave = np.array([LH_pet, HL_pet, HH_pet])
            PET_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            gt_wave = np.array([LH_gt, HL_gt, HH_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))    


def write_h5_12turnel(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,12,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,12,176,176),expectedrows=1000000)
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
        
        for j in range(PET.shape[2]-2):
            
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            coeffs_pet1 = pywt.dwt2(PET[:,:,j+1], wavelet)
            LL1_pet, (LH1_pet, HL1_pet, HH1_pet) = coeffs_pet1
            coeffs_pet2 = pywt.dwt2(PET[:,:,j+2], wavelet)
            LL2_pet, (LH2_pet, HL2_pet, HH2_pet) = coeffs_pet2                        
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet, LL1_pet, LH1_pet, HL1_pet, HH1_pet, LL2_pet, LH2_pet, HL2_pet, HH2_pet])
            PET_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            coeffs_gt1 = pywt.dwt2(gt[:,:,j+1], wavelet)
            LL1_gt, (LH1_gt, HL1_gt, HH1_gt) = coeffs_gt1
            coeffs_gt2 = pywt.dwt2(gt[:,:,j+2], wavelet)
            LL2_gt, (LH2_gt, HL2_gt, HH2_gt) = coeffs_gt2                      
            gt_wave = np.array([LL_gt, LH_gt, HL_gt, HH_gt, LL1_gt, LH1_gt, HL1_gt, HH1_gt, LL2_gt, LH2_gt, HL2_gt, HH2_gt])
            Ground_truth_pool.append(np.expand_dims(gt_wave, axis=0))
                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0)) 


def write_h5_3wave2(target_data_dir,dataset,target_file,index):
    start = 0
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,176,176),expectedrows=1000000)
    filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
                                               shape=(0,1),expectedrows=1000000)
    for name in index:
        PET_path = os.path.join(target_data_dir,'50_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
        
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            print('j:{}'.format(j))
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_pool.append(np.expand_dims(LH_pet, axis=0))
            PET_pool.append(np.expand_dims(HL_pet, axis=0))
            PET_pool.append(np.expand_dims(HH_pet, axis=0))
            print('PET_pool:{}'.format(PET_pool.shape))
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
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
        PET_path = os.path.join(target_data_dir,'50_dose',"patient_"+ str(name)+".nii")
        PET = np.array(nib.load(PET_path).dataobj)
        PET = PET/max_img
        PET = PET[(PET.shape[0]//2-176):(PET.shape[0]//2+176), (PET.shape[1]//2-176):(PET.shape[1]//2+176)]
        
        gt_path = os.path.join(target_data_dir,'Normal',"patient_"+ str(name)+".nii")
        gt = np.array(nib.load(gt_path).dataobj)
        gt = gt/max_img
        gt = gt[(gt.shape[0]//2-176):(gt.shape[0]//2+176), (gt.shape[1]//2-176):(gt.shape[1]//2+176)]
        
        for j in range(PET.shape[2]):
            print('j:{}'.format(j))
            coeffs_pet = pywt.dwt2(PET[:,:,j], wavelet)
            LL_pet, (LH_pet, HL_pet, HH_pet) = coeffs_pet
            PET_pool.append(np.expand_dims(LL_pet, axis=0))

            print('PET_pool:{}'.format(PET_pool.shape))
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
            Ground_truth_pool.append(np.expand_dims(LL_gt, axis=0))

                
            out_filename = "filenames" + '+{}+{}'.format(name, j)
            filenames_pool.append(np.expand_dims([out_filename], axis=0))

#path = "/media/data/fanxuan/data/PART2_Siemens"
path = "/media/data/fanxuan/data/PART2"
target_path = "/media/data/fanxuan/data"

n = 0
wavelet = 'haar'
#wavelet = 'bior3.7'
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
    target_file = tables.open_file(os.path.join(save_path, 'data_Siemens_normal_50_bio_3_7.h5'), mode='w')
    target_data_dir = '/media/data/fanxuan/data/PART2_Siemens'
    write_h5_4turnel_test(target_data_dir, 'valid', target_file, index)

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
    target_file = tables.open_file(os.path.join(save_path, 'data_wave_10_haar_last.h5'), mode='w')
    target_data_dir = '/media/data/fanxuan/data/PART2'
    write_h5_4turnel_last(target_data_dir, 'train', target_file, train_index)
    write_h5_4turnel_last(target_data_dir, 'valid', target_file, valid_index)

    target_file.close()
