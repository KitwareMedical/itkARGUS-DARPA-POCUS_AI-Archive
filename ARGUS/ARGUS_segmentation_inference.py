import configparser

import os
import json
import numpy as np

import itk
from itk import TubeTK as tube

import torch

from monai.networks.nets import UNet
from monai.transforms import ( Resize, ScaleIntensityRange, ToTensor )
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

from ARGUS_Transforms import *

class ARGUS_segmentation_inference:

    def __init__(self, config_file_name, network_name="final", device_num=0):
        
        config = configparser.ConfigParser()
        config.read(config_file_name)

        self.network_name = network_name
        if device_num >= 0:
            self.device = torch.device("cuda:" + str(device_num))
        else:
            self.device = "cpu"

        self.num_classes = int(config[network_name]['num_classes'])
        
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
            
        self.net_layer_channels = tuple([int(x) for x in json.loads(config[network_name]['layer_channels'])])
        self.net_layer_strides = tuple([int(x) for x in json.loads(config[network_name]['layer_strides'])])
        self.net_num_residual_units = int(config[network_name]['num_residual_units'])
        
        self.class_blur = [float(x) for x in json.loads(config[network_name]['class_blur'])]
        self.class_min_size = [int(x) for x in json.loads(config[network_name]['class_min_size'])]
        self.class_max_size = [int(x) for x in json.loads(config[network_name]['class_max_size'])]
        tmp_class_keep_only_largest = [int(x) for x in json.loads(config[network_name]['class_keep_only_largest'])]
        self.class_keep_only_largest = np.where(tmp_class_keep_only_largest==1, True, False)
        self.class_morph = [int(x) for x in json.loads(config[network_name]['class_morph'])]

        self.model = [UNet(
            dimensions=self.net_in_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
            channels=self.net_layer_channels,
            strides=self.net_layer_strides,
            num_res_units=self.net_num_residual_units,
            norm=Norm.BATCH,
            ).to(self.device)]*self.num_models
        
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
            axis=2)
        
        self.ConvertToTensor = ToTensor()
        
    def load_model(self, model_num, filename):
        self.model[model_num].load_state_dict(torch.load(filename, map_location=self.device))
        self.model[model_num].eval()

    def preprocess(vid):
        ar_input_array = np.empty([1, 1,
                               self.num_channels, vid.shape[0], vid.shape[1]])

        input_array = self.SaptialResize(vid)
        
        input_array_scaled = self.IntensityScale(input_array)
        
        ar_input_array[0, 0] = self.ARGUS_preprocess(input_array_scaled)
        
        self.input_tensor = self.ConvertToTensor(ar_input_array.astype(np.float32))

    def clean_probabilities(self, run_output, use_blur=True):
        if use_blur:
            prob = np.empty(run_output.shape)
            for c in range(self.num_classes):
                itkProb = itk.GetImageFromArray(run_output[c])
                imMathProb = tube.ImageMath.New(itkProb)
                imMathProb.Blur(self.class_blur[c])
                itkProb = imMathProb.GetOutput()
                prob[c] = itk.GetArrayFromImage(itkProb)
        else:
            prob = run_output.copy()
        pmin = prob.min()
        pmax = prob.max()
        prange = pmax - pmin
        prob = (prob - pmin) / prange
        for c in range(1,self.num_classes):
            class_array = np.argmax(prob, axis=0)
            class_array[:self.class_morph[c]*2,:] = 0
            class_array[:,:self.class_morph[c]*2] = 0
            class_array[-self.class_morph[c]*2:,:] = 0
            class_array[:,-self.class_morph[c]*2:] = 0
            count = np.count_nonzero(class_array == c)
            done = False
            op_iter = 0
            op_iter_max = 40
            while not done and op_iter < op_iter_max:
                done = True
                while count < self.class_min_size[c] and op_iter < op_iter_max:
                    prob[c] = prob[c] * 1.05
                    class_array = np.argmax(prob, axis=0)
                    class_array[:self.class_morph[c]*2,:] = 0
                    class_array[:,:self.class_morph[c]*2] = 0
                    class_array[-self.class_morph[c]*2:,:] = 0
                    class_array[:,-self.class_morph[c]*2:] = 0
                    count = np.count_nonzero(class_array == c)
                    op_iter += 1
                    done = False
                while count > self.class_max_size[c] and op_iter < op_iter_max:
                    prob[c] = prob[c] * 0.95
                    class_array = np.argmax(prob, axis=0)
                    class_array[:self.class_morph[c]*2,:] = 0
                    class_array[:,:self.class_morph[c]*2] = 0
                    class_array[-self.class_morph[c]*2:,:] = 0
                    class_array[:,-self.class_morph[c]*2:] = 0
                    count = np.count_nonzero(class_array == c)
                    op_iter += 1
                    done = False
        print("Iterations to optimize prior =", op_iter)
        denom = np.sum(prob, axis=0)
        denom = np.where(denom == 0, 1, denom)
        prob =  prob / denom

        return prob

    def classify_probabilities(self, prob):
        class_array = np.argmax(prob, axis=0)
        class_array[:self.class_morph[1]*2,:] = 0
        class_array[:,:self.class_morph[1]*2] = 0
        class_array[-self.class_morph[1]*2:,:] = 0
        class_array[:,-self.class_morph[1]*2:] = 0
        class_image = itk.GetImageFromArray(
            class_array.astype(np.float32))

        #itk.imwrite(
            #itk.GetImageFromArray(prob_total.astype(np.float32)),
            #"prob_total_f" + str(self.vfold_num) +
            #"i" + str(image_num) + ".mha")
        #itk.imwrite(class_image,
            #"class_image_init_f" + str(self.vfold_num) +
            #"i" + str(image_num) + ".mha")

        for c in range(1,self.num_classes):
            imMathClassCleanup = tube.ImageMath.New(class_image)
            if self.class_morph[c] > 0:
                imMathClassCleanup.Dilate(self.class_morph[c], c, 0)
                imMathClassCleanup.Erode(self.class_morph[c], c, 0)
                class_array[:2*self.class_morph[1],:] = 0
                class_array[:,:2*self.class_morph[1]] = 0
                class_array[-2*self.class_morph[1]:,:] = 0
                class_array[:,-2*self.class_morph[1]:] = 0
            imMathClassCleanup.Threshold(c, c, 1, 0)
            class_clean_image = imMathClassCleanup.GetOutputUChar()

            if self.class_keep_only_largest[c]==1:
                seg = itk.itkARGUS.SegmentConnectedComponents.New(
                    Input=class_clean_image
                )
                seg.SetKeepOnlyLargestComponent(True)
                seg.Update()
                class_clean_image = seg.GetOutput()

            class_clean_array = itk.GetArrayFromImage(class_clean_image)
            class_array = np.where(class_array == c, 0, class_array)
            class_array = np.where(class_clean_array != 0, c, class_array)

        return class_array
    
    def inference():
        roi_size = (self.size_x, self.size_y)
        prob_size = (self.num_classes, self.size_x, self.size_y)
        prob_total = np.zeros(prob_size)
        with torch.no_grad():
            for m in range(self.num_models):
                test_outputs = sliding_window_inference(
                    self.input_tensor.to(self.device), roi_size, 1, self.model[model_num])
                prob = self.clean_probabilities(test_outputs)
                prob_total += prob
        prob_total /= self.num_models
        prob = self.clean_probabilities(prob_total, use_blur=False)
        class_array = self.classify_probabilities(prob_total)
    
        return class_array