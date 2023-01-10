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
        
    def generate_roi(self, ar_image, ar_array, ar_labels):
        print(ar_labels.shape)
        print(ar_labels.shape[1])
        print(ar_labels.shape[0])
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
        print(ar_image_size)
        print(roi_min_x, roi_min_y, 0)
        print(roi_max_x, roi_max_y, ar_image_size[2])
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
        
        self.input_tensor = self.ConvertToTensor(roi_input_array)
        self.label_tensor = self.ConvertToTensor(roi_label_array)
        
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