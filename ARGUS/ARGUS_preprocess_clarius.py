import os
import glob

import numpy as np
from os import path

import itk
from itk import TubeTK as tube

class ARGUS_preprocess_clarius():

    def __init__(self, new_size):
        ImageFloat = itk.Image[itk.F,3]
        
        preload = tube.CropImage[ImageFloat,ImageFloat].New()
        preload = itk.ResampleImageFilter[ImageFloat, ImageFloat].New()
        
        self.new_size = new_size
        
    def get_ruler_points(self, img):
        """ Find points along ruler on right side of image """
        mid_z = img.shape[0]//2
        
        y_min = 10
        y_max = img.shape[1]-10
        
        min_x = 2522
        max_x = 2524
        mid = np.mean((img[mid_z,y_min:y_max,min_x:max_x]//200),axis=1)
        y = np.flatnonzero(mid)+y_min
        return y

    def get_roi(self, img):
        y = self.get_ruler_points(img)
        assert len(y) > 10, "Could not find ruler in Clarius fromat."
        
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

        assert len(y) > 5, "Could not find ruler in Clarius fromat."

        avg += y[y.size-1]
        count += 1
        avg /= count
        yCenters.append(avg)
        avg = 0
        for j in range(len(yCenters)-1):
            avg += yCenters[j+1]-yCenters[j]

        assert len(yCenters) > 5, "Could not find ruler in Clarius fromat."

        avg /= len(yCenters)-1
    
        tic_num = len(yCenters)
        tic_min = yCenters[0]
        tic_max = yCenters[-1]
        tic_diff = avg
        return tic_num,int(tic_min),int(tic_max),tic_diff

    def process(self, vid):
        
        vid_array = itk.GetArrayViewFromImage(vid)
        tic_num,tic_min,tic_max,tic_diff = self.get_roi(vid_array)

        pixel_spacing = 2/tic_diff
        spacing = [pixel_spacing,pixel_spacing,vid.GetSpacing()[2]]
        vid.SetSpacing(spacing)

        crop_min_z = 0
        crop_max_z = vid.shape[0]
        mid_z = crop_max_z//2
        
        crop_min_y = int(tic_min+tic_diff)
        crop_max_y = int(tic_max-tic_diff)
        
        crop_min_x = 1255
        crop_max_x = 2510
        
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
        new_sp = [(sz[0]*sp[0])/self.new_size[0],
                  (sz[1]*sp[1])/self.new_size[1],
                  sp[2]]
        
        if new_sp[0]/sp[0] > 2:
            ImMath = tube.ImageMath.New(tmp_new_img)
            ImMath.BlurOrder(new_sp[0]/3, 0, 0)
            ImMath.BlurOrder(new_sp[1]/3, 0, 1)
            tmp_new_img = ImMath.GetOutput()
            
        interpolator = itk.LinearInterpolateImageFunction.New(tmp_new_img)
        Resample = itk.ResampleImageFilter.New(Input=tmp_new_img)
        Resample.SetInterpolator(interpolator)
        Resample.SetOutputStartIndex([0,0,0])
        Resample.SetOutputSpacing(new_sp)
        Resample.SetOutputOrigin(new_org)
        Resample.SetSize([self.new_size[0], self.new_size[1], sz[2]])
        Resample.SetOutputDirection(tmp_new_img.GetDirection())
        Resample.Update()
        img = Resample.GetOutput()

        return img
