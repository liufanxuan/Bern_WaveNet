import os
import numpy as np
import dicom2nifti
import shutil
import tables
import nibabel as nib
from skimage import measure

def compute_metrics(real_input, pred_input):
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
	print('ssim:{}'.format(ssim))

	return mse, nrmse, mape, psnr, ssim

normal_path = '/media/data/fanxuan/data/PART2/Normal/patient_13.nii'
gen_path = '/media/data/fanxuan/data/PART2/10_dose/patient_13.nii'
gen_data = np.array(nib.load(gen_path).dataobj)
real_data = np.array(nib.load(normal_path).dataobj)
compute_metrics(real_data, gen_data)