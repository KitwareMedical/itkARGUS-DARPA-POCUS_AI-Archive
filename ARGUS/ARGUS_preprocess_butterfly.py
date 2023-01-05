import os
import glob

import numpy as np
from os import path

import itk
from itk import TubeTK as tube

class ARGUS_preprocess_butterfly():

    def __init__(self):
        ImageFloat = itk.Image[itk.F,3]
        self.Crop = tube.CropImage[ImageFloat,ImageFloat].New()
        self.Change = itk.ChangeInformationImageFilter[ImageFloat].New()
        self.Resample = tube.ResampleImage[ImageFloat].New()
        
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
        tic_num,tic_min,tic_max,tic_diff = self.get_roi(vid)

        pixels_per_unit = 40.92
        scale = tic_diff / pixels_per_unit
        center_x = int((780-120)/2+120)
        offset_x = center_x-120
        crop_min_x = 120
        crop_max_x = 780
        crop_min_y = int(tic_min+tic_diff)
        crop_max_y = int(tic_max-tic_diff)
        crop_min_z = 0
        crop_max_z = vid.shape[0]
        self.Crop.SetInput(vid)
        self.Crop.SetMin([crop_min_x,crop_min_y,crop_min_z])
        self.Crop.SetMax([crop_max_x,crop_max_y,crop_max_z])
        self.Crop.Update()
        tmp_crop_img = self.Crop.GetOutput()

        pixel_spacing = 2/pixels_per_unit
        spacing = [pixel_spacing,pixel_spacing,pixel_spacing]
        origin = [4,0,0]
        self.Change.SetInput(tmp_crop_img)
        self.Change.SetOutputSpacing(spacing)
        self.Change.ChangeSpacingOn()
        self.Change.SetOutputOrigin(origin)
        self.Change.ChangeOriginOn()
        self.Change.UpdateOutputInformation()
        self.Change.Update()
        tmp_new_img = self.Change.GetOutput()

        #for Spacing and Size to use the same axis ordering, don't use img.shape\n",
        sz = list(tmp_new_img.GetLargestPossibleRegion().GetSize())
        sp = list(tmp_new_img.GetSpacing())
        sp[0] = sz[0]/320.25*sp[0]
        sp[1] = sz[1]/320.25*sp[1]
        self.Resample.SetInput(tmp_new_img)
        self.Resample.SetInterpolator("NearestNeighbor")
        self.Resample.SetSpacing(sp)
        self.Resample.Update()
        img = self.Resample.GetOutput()

        return img