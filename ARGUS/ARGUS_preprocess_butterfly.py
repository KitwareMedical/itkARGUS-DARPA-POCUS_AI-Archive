import os
import glob

import numpy as np
from os import path

import itk
from itk import TubeTK as tube

class ARGUS_preprocess_butterfly():

    def __init__(self, new_size=None):
        ImageFloat = itk.Image[itk.F,3]
        
        preload = tube.CropImage[ImageFloat,ImageFloat].New()
        preload = itk.ResampleImageFilter[ImageFloat, ImageFloat].New()
        
        self.new_size = new_size
        
    def get_ruler_points(self, img):
        """ Find points along ruler on right side of image """
        mid_z = img.shape[0]//2
        
        y_min = 40
        y_max = 1080
        
        max_x = img.shape[2]-1
        count = np.count_nonzero(img[mid_z,y_min:y_max,max_x]//50)
        while max_x > img.shape[2]-20 and count <= 12:
            max_x -= 1
            count = np.count_nonzero(img[mid_z,y_min:y_max,max_x]//50)
        min_x = max_x - 1
        count = np.count_nonzero(img[mid_z,y_min:y_max,min_x]//50)
        while min_x > img.shape[2]-25 and count >= 10:
            min_x -= 1
            count = np.count_nonzero(img[mid_z,y_min:y_max,min_x]//50)
        mid_x = (max_x * 3 + min_x) // 4
        min_x = mid_x-1
        max_x = mid_x+1
        #min_x = img.shape[2] - 4 #907
        #max_x = img.shape[2] - 5 #908
        mid = np.mean((img[mid_z,y_min:y_max,min_x:max_x]//50),axis=1)
        y = np.flatnonzero(mid)+y_min
        return y

    def get_roi(self, img):
        y = self.get_ruler_points(img)

        assert len(y) > 5, "Could not find ruler in Butterfly format."
        assert len(y) < 75, "Could not find ruler in Butterfly format."
        
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

        assert len(yCenters) > 5, "Could not find ruler in Butterfly format."
        assert len(yCenters) < 75, "Could not find ruler in Butterfly format."

        avg = 0
        for j in range(len(yCenters)-1):
            avg += yCenters[j+1]-yCenters[j]

        assert len(yCenters) > 5, "Could not find ruler in Butterfly format."
        assert len(yCenters) < 75, "Could not find ruler in Butterfly format."

        avg /= len(yCenters)-1
    
        tic_num = len(yCenters)
        tic_min = yCenters[0]
        tic_max = yCenters[-1]
        tic_diff = avg
        return tic_num,int(tic_min),int(tic_max),tic_diff

    def process(self, vid, new_size=None):
        
        if new_size != None:
            self.new_size = new_size
        elif self.new_size != None:
            new_size = self.new_size
        else:
            new_size = [320,320]
        
            
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
        
        mid_x = vid_array.shape[2]//2
        min_x = mid_x-10
        count = np.count_nonzero(vid_array[mid_z,:,min_x]//10)
        while min_x > 100 and count > 10:
            min_x -= 1
            count = np.count_nonzero(vid_array[mid_z,:,min_x]//10)
        max_x = mid_x + 1
        count = np.count_nonzero(vid_array[mid_z,:,max_x]//10)
        while max_x < vid_array.shape[2]-100 and count > 10:
            max_x += 1
            count = np.count_nonzero(vid_array[mid_z,:,max_x]//10)
        crop_min_x = min_x + 10
        crop_max_x = max_x - 10
        
        Crop = tube.CropImage.New(Input=vid)
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
        
        if new_sp[0]/sp[0] > 2:
            ImMath = tube.ImageMath.New(Input=tmp_new_img)
            ImMath.BlurOrder(new_sp[0]/3, 0, 0)
            ImMath.BlurOrder(new_sp[1]/3, 0, 1)
            tmp_new_img = ImMath.GetOutput()
        
        interpolator = itk.LinearInterpolateImageFunction.New(tmp_new_img)
        Resample = itk.ResampleImageFilter.New(Input=tmp_new_img)
        Resample.SetInterpolator(interpolator)
        Resample.SetSize([new_size[0], new_size[1], sz[2]])
        Resample.SetOutputStartIndex([0,0,0])
        Resample.SetOutputSpacing(new_sp)
        Resample.SetOutputOrigin(new_org)
        Resample.SetOutputDirection(tmp_new_img.GetDirection())
        Resample.Update()
        img = Resample.GetOutput()

        return img
