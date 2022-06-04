#!/usr/bin/env python
# coding: utf-8

import sys

assert len(sys.argv) == 3, 'usage: <inputdir_with_mha> <outputdir>'

import os

import glob

import numpy as np

import itk
from itk import TubeTK as tube

from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_dilation
from scipy.ndimage import binary_dilation


def get_ruler_points_Butterfly(img):
    """ Find points along ruler on right side of image """
    y_min = 40
    y_max = 1080
    min_x = 907
    max_x = 908
    mid = np.mean((img[0,y_min:y_max,min_x:max_x]//100),axis=1)
    y = np.flatnonzero(mid)+y_min
    return y


def get_data_roi_from_Butterfly(img):
    y = get_ruler_points_Butterfly(img)

    img_shape = img.shape

    avg = 0
    count = 0
    yCenters = []
    for j in range(y.size-1):
        avg += y[j]
        count += 1
        if(y[j+1]-y[j]>5):
            avg /= count
            yCenters.append(avg)
            avg = 0
            count = 0
    avg += y[y.size-1]
    count += 1
    avg /= count
    yCenters.append(avg)
    avg = 0
    for j in range(len(yCenters)-1):
        avg += yCenters[j+1]-yCenters[j]
    avg /= len(yCenters)-1    
    

    tic_num = len(yCenters)
    tic_min = yCenters[0]
    tic_max = yCenters[len(yCenters)-1]
    tic_diff = avg

    return tic_num,tic_min,tic_max,tic_diff


def crop_Butterfly(img):
    tic_num,tic_min,tic_max,tic_diff = get_data_roi_from_Butterfly(img)
    print(tic_num,'[',tic_min,'-',tic_max,']',tic_diff)
    
    pixels_per_unit = 40.92
    scale = tic_diff / pixels_per_unit
    center_x = int((780-120)/2+120)
    offset_x = center_x-120
    crop_min_x = 120
    crop_max_x = 780
    crop_min_y = int(tic_min+tic_diff)
    crop_max_y = int(tic_max-tic_diff)
    crop_min_z = 0
    crop_max_z = img.shape[0]
    Crop = tube.CropImage.New(img)
    Crop.SetMin([crop_min_x,crop_min_y,crop_min_z])
    Crop.SetMax([crop_max_x,crop_max_y,crop_max_z])
    Crop.Update()
    tmp_crop_img = Crop.GetOutput()
    
    pixel_spacing = 2/pixels_per_unit
    spacing = [pixel_spacing,pixel_spacing,pixel_spacing]
    origin = [4,0,0]
    Change = itk.ChangeInformationImageFilter.New(tmp_crop_img)
    Change.SetOutputSpacing(spacing)
    Change.ChangeSpacingOn()
    Change.SetOutputOrigin(origin)
    Change.ChangeOriginOn()
    Change.UpdateOutputInformation()
    Change.Update()
    new_img = Change.GetOutput()
    
    return new_img


def find_text_Butterfly(img):
    x_min = 600
    x_max = 850
    y_min = 700
    y_max = 1000
    arr_full3 = itk.GetArrayViewFromImage(img)
    
    arr3 = arr_full3[:,y_min:y_max,x_min:x_max]
    arr_diffY3 = np.diff(arr3, axis=2, append=0)
    arr_diffY3 = abs(np.diff(arr_diffY3, axis=2, append=0))
    arr_diffX3 = np.diff(arr3, axis=1, append=0)
    arr_diffX3 = abs(np.diff(arr_diffX3, axis=1, append=0))
    arr_sumZ_diffY = np.mean(arr_diffY3, axis=0)
    arr_sumZ_diffX = np.mean(arr_diffX3, axis=0)
    arr_diff = arr_sumZ_diffY + arr_sumZ_diffX
    arr_diff = gaussian_filter(arr_diff, sigma=2)
    arr_mask = np.where(arr_diff>40,1,0)
    arr_mask[:,-2:]=0
    arr_mask[-2:,:]=0
    arr_mask3 = np.repeat(arr_mask[np.newaxis,:,:], arr3.shape[0], axis=0)

    arr_hole3 = np.where(arr_mask3==0,arr3,0)
    arr_hole_fill3 = grey_dilation(arr_hole3, size=(2,20,20))
    
    arr_fix3 = np.where(arr_mask3==0,arr3,arr_hole_fill3)
    
    arr_maskD = binary_dilation(arr_mask,iterations=4)
    arr_maskD3 = np.repeat(arr_maskD[np.newaxis,:,:], arr3.shape[0], axis=0)
    arr_mixD3 = gaussian_filter(arr_fix3,sigma=3)
    
    arr_fix3 = np.where(arr_maskD3==0,arr3,arr_mixD3)
    
    arr_full3[:,y_min:y_max,x_min:x_max] = arr_fix3
    
    return img


def main():
    src_path = sys.argv[1]
    src_file_wildcard = os.path.join(src_path,"*.mha")
    dst_path = sys.argv[2]
    os.makedirs(dst_path, exist_ok=True)
    files = glob.glob(src_file_wildcard)
    num_files = len(files)
    for i,file in enumerate(files):
        tmp_img = itk.imread(file,itk.UC)
        
        fixed_img = find_text_Butterfly(tmp_img)
        new_img = crop_Butterfly(fixed_img)
        
        fpath,fname = os.path.split(file)
        fname_base,fname_ext = os.path.splitext(fname)
        new_name = os.path.join(dst_path,fname_base+'_cropM.mha')
        itk.imwrite(new_img,new_name,compression=True)
        print(fname,'>',new_name)

main()
