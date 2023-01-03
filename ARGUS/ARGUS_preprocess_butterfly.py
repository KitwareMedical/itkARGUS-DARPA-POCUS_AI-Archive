import os
import glob

import numpy as np
from os import path

import itk
from itk import TubeTK as tube

class ARGUS_preprocess_butterfly():

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

    def process(self, vid):
        tic_num,tic_min,tic_max,tic_diff = self.get_roi(img)
    
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