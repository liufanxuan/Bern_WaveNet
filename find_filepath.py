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
import  gzip

def find_filepath():
    filepath_list = []
    path = '/media/data/fanxuan/data/data_all_nii/'
    for root, dirs, files in os.walk(path):
        if '4_dose' in root:
            for niifile in files:
                filepath_list.append(os.path.join(root,niifile))
                if '.gz.gz' in niifile:
                    # os.remove(os.path.join(root,niifile))
                    print(niifile)
    return filepath_list
normal_path_list = find_filepath()
print(len(normal_path_list))
random.shuffle(normal_path_list)
file_name = os.path.basename(normal_path_list[1])
file_base = os.path.dirname(os.path.dirname(normal_path_list[1]))

#print(normal_path_list)
#print(file_name)
#print(file_base)


def compress_delete_file(data_file):

    gz_path = data_file+'.gz'
    if os.path.exists(gz_path):
        print('exist&delete:{}'.format(gz_path))
    '''if '.gz' not in data_file:
        #pf = open(gz_path, 'wb')
        #data = open(data_file,'rb').read()
        #data_comp = gzip.compress(data)
        #pf.write(data_comp)
        #pf.close()
        print('create:{}'.format(gz_path))
        if os.path.exists(data_file):
            print('delete:{}'.format(data_file))
        else:
            print('False!!!!!')'''
for data_file in normal_path_list:
    compress_delete_file(data_file)

'''
def compress_delete_file(data_file):
    gz_path = data_file+'.gz'
    pf = open(gz_path, 'wb')
    data = open(data_file,'rb').read()
    data_comp = gzip.compress(data)
    pf.write(data_comp)
    pf.close()
    
compress_delete_file(normal_path_list[1])

normal_path_list = '/media/data/fanxuan/data/data_all_nii/uExplorer/PART3/Normal/type1_part3_patient_11.nii'
zip_filename = normal_path_list+'.gz'

with open(zip_filename, 'rb') as pr, open('/media/data/fanxuan/data/data_all_nii/uExplorer/PART3/'+'new.nii','wb') as pw:
    pw.write(gzip.decompress(pr.read())  ) 
  
pet1 = np.array(nib.load('/media/data/fanxuan/data/data_all_nii/uExplorer/PART3/'+'new.nii').dataobj)
pet2 = np.array(nib.load(normal_path_list).dataobj)
print(np.mean(np.square(pet1-pet2)))


if os.path.exists('/media/data/fanxuan/data/data_all_nii/uExplorer/PART3/'+'new.nii'):
    os.remove('/media/data/fanxuan/data/data_all_nii/uExplorer/PART3/'+'new.nii')
else:
    print('False!!!!!')
'''