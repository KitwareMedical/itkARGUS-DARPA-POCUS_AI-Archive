import os
import glob

import numpy as np
from os import path

import itk
from itk import TubeTK as tube

class ARGUS_preprocess_butterfly():

    def __init__(self):
        ImageFloat = itk.Image[itk.F,3]
        
        preload = tube.CropImage[ImageFloat,ImageFloat].New()
        preload = itk.ResampleImageFilter[ImageFloat, ImageFloat].New()
        
    def get_ruler_points(self, img):
        """ Find points along ruler on right side of image """
        y_min = 40
        y_max = 1080
        min_x = 907
        max_x = 908
        mid = np.mean((img[0,y_min:y_max,min_x:max_x]//100),axis=1)
        y = np.flatnonzero(mid)+y_min
        return y

    def get_roi(self, img):
        y = self.get_ruler_points(img)
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

    def process(self, vid, new_size):
        
        tic_num,tic_min,tic_max,tic_diff = self.get_roi(vid)

        pixel_spacing = 2/tic_diff
        spacing = [pixel_spacing,pixel_spacing,vid.GetSpacing()[2]]
        vid.SetSpacing(spacing)

        center_x = int((780-120)/2+120)
        offset_x = center_x-120
        crop_min_x = 120
        crop_max_x = 780
        crop_min_y = int(tic_min+tic_diff)
        crop_max_y = int(tic_max-tic_diff)
        crop_min_z = 0
        crop_max_z = vid.shape[0]
        Crop = tube.CropImage.New(vid)
        Crop.SetMin([crop_min_x,crop_min_y,crop_min_z])
        Crop.SetMax([crop_max_x,crop_max_y,crop_max_z])
        Crop.Update()
        tmp_new_img = Crop.GetOutput()

        org = list(tmp_new_img.GetOrigin())
        indx = list(tmp_new_img.GetLargestPossibleRegion().GetIndex())
        sz = list(tmp_new_img.GetLargestPossibleRegion().GetSize())
        sp = list(tmp_new_img.GetSpacing())
        new_org = [indx[0]*sp[0], indx[1]*sp[1], org[2]] 
        new_sp = [(sz[0]*sp[0])/new_size[0], (sz[1]*sp[1])/new_size[1], sp[2]]
        Resample = itk.ResampleImageFilter.New(tmp_new_img)
        Resample.SetSize([new_size[0], new_size[1], sz[2]])
        Resample.SetOutputStartIndex([0,0,0])
        Resample.SetOutputSpacing(new_sp)
        Resample.SetOutputOrigin(new_org)
        Resample.SetOutputDirection(tmp_new_img.GetDirection())
        Resample.Update()
        img = Resample.GetOutput()

        return img