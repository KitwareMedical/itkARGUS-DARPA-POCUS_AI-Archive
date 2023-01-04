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
    ScaleIntensityRange,
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
        
        self.roi_class  = int(config[network_name]['roi_class'])
        
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
        
        self.SpatialResize =  Resize(
            spatial_size=[self.size_y,self.size_x],
            mode='bilinear')
            
        self.IntensityScale = ScaleIntensityRange(
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0)
        
        self.ARGUS_preprocess = ARGUS_RandSpatialCropSlices(
            num_slices=self.num_slices,
            center_slice=self.testing_slice,
            reduce_to_statistics=self.reduce_to_statistics,
            extended=self.reduce_to_statistics,
            include_center_slice=self.reduce_to_statistics,
            include_gradient=self.reduce_to_statistics,
            axis=0)
        
        self.ConvertToTensor = ToTensor()

    def init_model(self, model_num):
        self.model[model_num] = monai.networks.nets.DenseNet121(
            spatial_dims=self.net_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
        ).to(self.device)
        
    def load_model(self, model_num, filename):
        self.model[model_num].load_state_dict(torch.load(filename, map_location=self.device))
        self.model[model_num].eval()
        
    def preprocess_using_ar_data(self, input_array, ar_labels):
        roi_min_x = 0
        roi_max_x = input_array.shape[1]-1
        while( np.count_nonzero(ar_labels[:, roi_min_x, :]==self.roi_class)==0
               and roi_min_x<roi_max_x ):
            roi_min_x += 1
        while( np.count_nonzero(ar_labels[:, roi_max_x, :]==self.roi_class)==0
               and roi_max_x>roi_min_x):
            roi_max_x -= 1
        roi_mid_x = (roi_min_x + roi_max_x)//2
        roi_min_x = max(roi_mid_x-self.size_x//2, 0)
        roi_max_x = min(roi_min_x+self.size_x, image.shape[1]-1)
        roi_min_x = roi_max_x-self.size_x
        
        roi_min_y = 0
        roi_max_y = input_array.shape[2]-1
        while( np.count_nonzero(ar_labels[:, :, roi_min_y]==self.roi_class)==0
               and roi_min_y<roi_max_y ):
            roi_min_y += 1
        while( np.count_nonzero(ar_labels[:, :, roi_max_y]==self.roi_class)==0
               and roi_max_y>roi_min_y):
            roi_max_y -= 1
        roi_mid_y = (roi_min_y + roi_max_y)//2
        roi_min_y = max(roi_mid_y-self.size_y//2, 0)
        roi_max_y = min(roi_min_y+self.size_y, image.shape[2]-1)
        roi_min_y = roi_max_y-self.size_y
    
        roi_input_array = input_array[:, roi_min_x:roi_max_x, roi_min_y:roi_max_y]
        
        self.input_tensor = self.ConvertToTensor(roi_input_array.astype(np.float32))
        
        return self.input_tensor
        
    def preprocess(self, vid):
        roi_input_array = np.empty([1, 1,
            self.net_in_channels, vid.shape[1], vid.shape[2]])

        tmp_testing_slice = self.testing_slice
        if tmp_testing_slice < 0:
            tmp_testing_slice = vid.shape[0]+tmp_testing_slice-1
        min_slice = max(0,tmp_testing_slice-self.num_slices//2-1)
        max_slice = min(vid.shape[0],tmp_testing_slice+self.num_slices//2+2)
        print(min_slice, max_slice)
        vid_roi = vid[min_slice:max_slice,:,:]
        
        input_array = self.SpatialResize(vid_roi)
        
        input_array_scaled = self.IntensityScale(input_array)
        print("input array scales shape =", input_array_scaled.shape)
        
        self.ARGUS_preprocess.center_slice = self.num_slices//2
        roi_input_array[0, 0] = self.ARGUS_preprocess(input_array_scaled).transpose([0,2,1])
        
        self.input_tensor = self.ConvertToTensor(roi_input_array.astype(np.float32))
        
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
    
    def inference(self):
        prob_total = np.zeros(prob_shape)
        with torch.no_grad():
            for run_num in range(num_runs):
                run_output = model(self.input_tensor[0].to(self.device))
                prob = self.clean_probabilities(run_output.cpu())
                prob_total += prob
        prob_total /= num_runs
        prob = self.clean_probabilities(prob_total)
        class_array = self.classify_probabilities(prob_total)
        return class_array, prob