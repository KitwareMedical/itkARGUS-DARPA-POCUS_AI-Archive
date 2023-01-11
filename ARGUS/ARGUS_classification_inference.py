import warnings
warnings.filterwarnings("ignore")

import os
import json
from glob import glob

import configparser

import random

import pathlib

import numpy as np

import itk
from itk import TubeTK as tube

import torch

import monai
from monai.transforms import (
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    Resize,
    ToTensor,
)

from ARGUS_Transforms import *

class ARGUS_classification_inference:
    def __init__(self, config_file_name, network_name="final", device_num=0):
 
        config = configparser.ConfigParser()
        config.read(config_file_name)

        self.network_name = network_name
        if device_num >= 0:
            self.device = torch.device("cuda:" + str(device_num))
        else:
            self.device = "cpu"

        self.num_classes  = int(config[network_name]['num_classes'])
        
        self.ar_roi_class  = int(config[network_name]['ar_roi_class'])
        
        self.num_slices = int(config[network_name]['num_slices'])
        self.size_x = int(config[network_name]['size_x'])
        self.size_y = int(config[network_name]['size_y'])
        
        self.testing_slice = int(config[network_name]['testing_slice'])
        
        self.num_models = int(config[network_name]['num_models'])
        
        self.net_in_dims = int(config[network_name]['num_input_dims'])
        
        self.net_in_channels = 1
        reduce_to_statistics_str = config[network_name]['reduce_to_statistics']
        self.reduce_to_statistics = False
        if( reduce_to_statistics_str == "True"):
            self.reduce_to_statistics = True   
            self.net_in_channels = 12
            
        self.model = [monai.networks.nets.DenseNet121(
            spatial_dims=self.net_in_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
        ).to(self.device)] * self.num_models
        
        ImageF = itk.Image[itk.F, 3]
        ImageSS = itk.Image[itk.SS, 3]
        crop = tube.CropImage[ImageF,ImageF].New()
        crop = tube.CropImage[ImageSS,ImageSS].New()
        resample = tube.ResampleImage[ImageF].New()
        resample = tube.ResampleImage[ImageSS].New()
        permute = itk.PermuteAxesImageFilter[ImageF].New()
        permute = itk.PermuteAxesImageFilter[ImageSS].New()
        
        self.ImageMath3F = tube.ImageMath[ImageF].New()
        
        ImageF2 = itk.Image[itk.F, 2]
        self.ImageMathF2 = tube.ImageMath[ImageF2].New()
        
        ImageS2 = itk.Image[itk.SS, 2]
        self.ImageMathS2 = tube.ImageMath[ImageS2].New()
        
        self.ARGUS_Preprocess = ARGUS_RandSpatialCropSlices(
            num_slices=self.num_slices,
            center_slice=self.testing_slice,
            reduce_to_statistics=self.reduce_to_statistics,
            extended=self.reduce_to_statistics,
            include_center_slice=self.reduce_to_statistics,
            include_gradient=self.reduce_to_statistics,
            axis=0)
        
        self.ARGUS_PreprocessLabel = ARGUS_RandSpatialCropSlices(
            num_slices=1,
            center_slice=self.testing_slice,
            reduce_to_statistics=False,
            extended=False,
            include_center_slice=True,
            include_gradient=False,
            axis=0)
        
        self.ConvertToTensor = ToTensor()
        
    def init_model(self, model_num):
        self.model[model_num] = monai.networks.nets.DenseNet121(
            spatial_dims=self.net_in_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
        ).to(self.device)
        
    def load_model(self, model_num, filename):
        self.model[model_num].load_state_dict(torch.load(filename, map_location=self.device))
        self.model[model_num].eval()
        
    def generate_roi(self, ar_image, ar_array, ar_labels):
        roi_min_x = 0
        roi_max_x = ar_labels.shape[1]-1
        while( np.count_nonzero(ar_labels[:, roi_min_x]==self.ar_roi_class)==0
               and roi_min_x<roi_max_x-1):
            roi_min_x += 1
        while( np.count_nonzero(ar_labels[:, roi_max_x]==self.ar_roi_class)==0
               and roi_max_x>roi_min_x+1):
            roi_max_x -= 1
        roi_mid_x = (roi_min_x + roi_max_x)//2
        roi_min_x = max(roi_mid_x-self.size_x//2, 0)
        roi_max_x = min(roi_min_x+self.size_x, ar_labels.shape[1]-1)
        roi_min_x = roi_max_x-self.size_x
        
        roi_min_y = 0
        roi_max_y = ar_labels.shape[0]-1
        while( np.count_nonzero(ar_labels[roi_min_y, :]==self.ar_roi_class)==0
               and roi_min_y<roi_max_y-1):
            roi_min_y += 1
        while( np.count_nonzero(ar_labels[roi_max_y, :]==self.ar_roi_class)==0
               and roi_max_y>roi_min_y+1):
            roi_max_y -= 1
        roi_mid_y = (roi_min_y + roi_max_y)//2
        roi_min_y = max(roi_mid_y-self.size_y//2, 0)
        roi_max_y = min(roi_min_y+self.size_y, ar_labels.shape[0]-1)
        roi_min_y = roi_max_y-self.size_y
    
        ar_image_size = ar_image.GetLargestPossibleRegion().GetSize()
        crop = tube.CropImage.New(Input=ar_image)
        crop.SetMin([roi_min_x, roi_min_y, 0])
        crop.SetMax([roi_max_x, roi_max_y, ar_image_size[2]])
        crop.Update()
        self.input_image = crop.GetOutput()
        
        self.input_array = ar_array[:, roi_min_y:roi_max_y, roi_min_x:roi_max_x]
        roi_input_array = np.empty([1, 1,
            self.net_in_channels, self.size_y, self.size_x])
        roi_input_array[0, 0] = self.input_array
        
        self.label_array = ar_labels[roi_min_y:roi_max_y, roi_min_x:roi_max_x]
        roi_label_array = np.empty([1, 1, self.size_y, self.size_x])
        roi_label_array[0, 0] = self.label_array
        
        self.input_tensor = self.ConvertToTensor(roi_input_array.astype(np.float32))
        self.label_tensor = self.ConvertToTensor(roi_label_array)
        
    def init_model(self, model_num):
        self.model[model_num] = UNet(
            dimensions=self.net_in_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
            channels=self.net_layer_channels,
            strides=self.net_layer_strides,
            num_res_units=self.net_num_residual_units,
            norm=Norm.BATCH,
            ).to(self.device)
        
    def load_model(self, model_num, filename):
        self.model[model_num].load_state_dict(torch.load(filename, map_location=self.device))
        self.model[model_num].eval()

    def preprocess(self, vid_img, lbl_img=None, slice_num=None, scale_data=True, rotate_data=True):
        ImageF = itk.Image[itk.F, 3]
        ImageSS = itk.Image[itk.SS, 3]
        
        img_size = vid_img.GetLargestPossibleRegion().GetSize()
        
        if slice_num != None:
            tmp_testing_slice = slice_num
        else:
            tmp_testing_slice = self.testing_slice
        if tmp_testing_slice < 0:
            tmp_testing_slice = img_size[2]+tmp_testing_slice-1
        min_slice = max(0,tmp_testing_slice-self.num_slices//2-1)
        max_slice = min(img_size[2],tmp_testing_slice+self.num_slices//2+2)
        
        min_index = [0, 0, min_slice]
        max_index = [img_size[0], img_size[1], max_slice]
        crop = tube.CropImage[ImageF,ImageF].New()
        crop.SetInput(vid_img)
        crop.SetMin(min_index)
        crop.SetMax(max_index)
        crop.Update()
        vid_roi_img = crop.GetOutput()
        if lbl_img != None:
            min_index = [0, 0, tmp_testing_slice]
            max_index = [img_size[0], img_size[1], tmp_testing_slice+1]
            crop = tube.CropImage[ImageSS,ImageSS].New()
            crop.SetInput(lbl_img)
            crop.SetMin(min_index)
            crop.SetMax(max_index)
            crop.Update()
            lbl_roi_img = crop.GetOutput()
            #lbl[tmp_testing_slice:tmp_testing_slice+1,:,:]
        else:
            lbl_roi_img = None
        
        resample = tube.ResampleImage[ImageF].New()
        resample.SetInput(vid_roi_img)
        size = [self.size_x, self.size_y, self.num_slices]
        resample.SetSize(size)
        resample.Update()
        vid_roi_img = resample.GetOutput()
        if lbl_img != None:
            resample = tube.ResampleImage[ImageSS].New()
            resample.SetInput(lbl_roi_img)
            size = [self.size_x, self.size_y, 1]
            resample.SetSize(size)
            resample.Update()
            lbl_roi_img = resample.GetOutput()
        
        if scale_data:
            self.ImageMath3F.SetInput(vid_roi_img)
            self.ImageMath3F.IntensityWindow(0,255,0,1)
            vid_roi_img = self.ImageMath3F.GetOutput()

        if rotate_data:
            permute = itk.PermuteAxesImageFilter[ImageF].New()
            permute.SetInput(vid_roi_img)
            order = [1,0,2]
            permute.SetOrder(order)
            permute.Update()
            vid_roi_img = permute.GetOutput()
            if lbl_img != None:
                permute = itk.PermuteAxesImageFilter[ImageSS].New()
                permute.SetInput(lbl_roi_img)
                order = [1,0,2]
                permute.SetOrder(order)
                permute.Update()
                lbl_roi_img = permute.GetOutput()
                
        vid_roi_array = itk.GetArrayFromImage(vid_roi_img)
        self.ARGUS_Preprocess.center_slice = self.num_slices//2
        roi_array = self.ARGUS_Preprocess(vid_roi_array)
        
        if lbl_img != None:
            lbl_array = itk.GetArrayFromImage(lbl_roi_img)[0]
        else:
            lbl_array = np.zeros([self.size_x, self.size_y])
        
        ar_input_array = np.empty([1,
                                   1,
                                   self.net_in_channels,
                                   self.size_x,
                                   self.size_y])
        ar_input_array[0,0] = roi_array
        
        ar_lbl_array = np.empty([1, 1, self.size_x, self.size_y])
        ar_lbl_array[0,0] = lbl_array
            
        self.input_image = vid_roi_img
        self.input_array = roi_array
        self.label_array = lbl_array
        
        self.input_tensor = self.ConvertToTensor(ar_input_array.astype(np.float32))
        self.label_tensor = self.ConvertToTensor(ar_lbl_array.astype(np.short))
        
        
    def clean_probabilities(self, run_output):
        prob = run_output.copy()
        pmin = prob.min()
        pmax = prob.max()
        prange = pmax - pmin
        prob = (prob - pmin) / prange
        denom = np.sum(prob, axis=0)
        denom = np.where(denom == 0, 1, denom)
        prob =  prob / denom
        return prob
    
    def classify_probabilities(self, run_output):
        class_num = np.argmax(run_output, axis=0)
        return class_num
    
    def inference(self)
        prob_total = np.zeros(self.num_classes)
        with torch.no_grad():
            for run_num in range(self.num_models):
                run_output = self.model[run_num](self.input_tensor[0].to(self.device))
                prob = self.clean_probabilities(run_output[0].cpu().detach().numpy())
                prob_total += prob
        prob_total /= self.num_models
        prob = self.clean_probabilities(prob_total)
        classification = self.classify_probabilities(prob_total)
        return classification, prob