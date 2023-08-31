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
import gzip

import pywt
import pywt.data

#max_img=498879.6
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
    base_name = os.path.basename(os.path.dirname(path_basic))
    print('base_name:')
    print(base_name)
    if "Full" in fold_name or "NORMAL" in fold_name or "normal" in fold_name:
        print('Full_fold_name:')
        print(fold_name)
        target_path = os.path.join(target_path,base_name, "Normal" )
    if "1-10 dose" in fold_name or "D10" in fold_name or "DRF_10" in fold_name:
        if "D100" not in fold_name:
            target_path = os.path.join(target_path, base_name, "10_dose" )
    if "1-20 dose" in fold_name or "D20" in fold_name or "DRF_20" in fold_name:
        target_path = os.path.join(target_path, base_name,"20_dose" )
    if "1-50 dose" in fold_name or "D50" in fold_name or "DRF_50" in fold_name:
        target_path = os.path.join(target_path, base_name,"50_dose" )
    if "1-100 dose" in fold_name or "D100" in fold_name or "DRF_100" in fold_name:
        target_path = os.path.join(target_path, base_name,"100_dose" )
    if "1-2 dose" in fold_name or "D2" in fold_name or "DRF_2" in fold_name:
        if "D20" not in fold_name:
            target_path = os.path.join(target_path, base_name,"2_dose" )
    if "1-4 dose" in fold_name or "D4" in fold_name or "DRF_4" in fold_name:
        target_path = os.path.join(target_path, base_name,"4_dose" )
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    nii_path =  os.path.join(target_path, "ori.nii" )
    print(path_basic)
    print(nii_path)

    dcm2nii(path_basic, nii_path)
    compress_delete_file(nii_path)
      
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
        PET_path = os.path.join(target_data_dir,'20_dose',"patient_"+ str(name)+".nii")
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
    PET_pool = target_file.create_earray(target_file.root,dataset+'_reduce',tables.Float32Atom(),
                                         shape=(0,4,176,176),expectedrows=1000000)
    Ground_truth_pool = target_file.create_earray(target_file.root,dataset+'_normal',tables.Float32Atom(),
                                                  shape=(0,4,176,176),expectedrows=1000000)
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
            PET_wave = np.array([LL_pet, LH_pet, HL_pet, HH_pet])
            PET_pool.append(np.expand_dims(PET_wave, axis=0))
            
            coeffs_gt = pywt.dwt2(gt[:,:,j], wavelet)
            LL_gt, (LH_gt, HL_gt, HH_gt) = coeffs_gt
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

'''
path = "/media/data/fanxuan/data/Subject_7_12"
target_path = "/media/data/fanxuan/data"
'''
n = -1
wavelet = 'haar'

path = '/media/data/fanxuan/data/Explorer_raw/'
target_path = '/media/data/fanxuan/data/Explorer_test_nii'
for root, dirs, files in os.walk(path):
    '''if files == [] and 'Anonymous_' in root:

        print('root:{} n:{}'.format(root, n))'''
    if dirs == [] and 'Anonymous_' in root:
        n = n+1
        print('file:{} n:{}'.format(root,n))
        file_to_nii(root, n, target_path)


'''index = np.arange(1,len([x for x in os.listdir(path) if not x.startswith('.')])+1)
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


index = np.arange(1,len([x for x in os.listdir("/media/data/fanxuan/data/PART2/Normal") if not x.startswith('.')])+1)
cross_validation = False
if cross_validation == False:
    train_index = index[:int(0.8 * len(index))]
    valid_index = index[int(0.8 * len(index)):]
    save_path = os.path.join('/media/data/fanxuan/data', os.path.basename(path)+'_{}'.format("h5data"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    target_file = tables.open_file(os.path.join(save_path, 'data_20.h5'), mode='w')
    target_data_dir = '/media/data/fanxuan/data/PART2'
    write_h5(target_data_dir, 'train', target_file, train_index)
    write_h5(target_data_dir, 'valid', target_file, valid_index)
    target_file.close()'''
