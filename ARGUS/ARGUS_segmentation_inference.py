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
        if device_num != None and device_num >= 0:
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
        
        if config.has_option(network_name, 'class_prior'):
            self.class_prior = [float(x) for x in json.loads(config[network_name]['class_prior'])]
        else:
            self.class_prior = np.ones([self.num_classes])
            
        self.class_blur = [float(x) for x in json.loads(config[network_name]['class_blur'])]
        self.class_min_size = [int(x) for x in json.loads(config[network_name]['class_min_size'])]
        self.class_max_size = [int(x) for x in json.loads(config[network_name]['class_max_size'])]
        self.class_keep_only_largest = [bool(x) for x in json.loads(config[network_name]['class_keep_only_largest'])]
        self.class_morph = [int(x) for x in json.loads(config[network_name]['class_morph'])]

        self.model = [UNet(
            spatial_dims=self.net_in_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
            channels=self.net_layer_channels,
            strides=self.net_layer_strides,
            num_res_units=self.net_num_residual_units,
            norm=Norm.BATCH,
        ).to(self.device)] * self.num_models
        
        # Preload these definitions
        ImageF = itk.Image[itk.F, 3]
        ImageSS = itk.Image[itk.SS, 3]
        pre_load = tube.CropImage[ImageF,ImageF].New()
        pre_load = tube.CropImage[ImageSS,ImageSS].New()
        pre_load = tube.ResampleImage[ImageF].New()
        pre_load = tube.ResampleImage[ImageSS].New()
        pre_load = itk.PermuteAxesImageFilter[ImageF].New()
        pre_load = itk.PermuteAxesImageFilter[ImageSS].New()
        
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
        self.ARGUS_Preprocess._gradient_cache = None
        self.ARGUS_Preprocess.cache_gradient = False
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
        
    def clean_probabilities_array(self, run_output, use_blur=True):
        if use_blur:
            prob = np.empty(run_output.shape)
            for c in range(self.num_classes):
                itkProb = itk.GetImageFromArray(run_output[c])
                self.ImageMathF2.SetInput(itkProb)
                if self.class_blur[c] > 0:
                    self.ImageMathF2.Blur(self.class_blur[c])
                itkProb = self.ImageMathF2.GetOutput()
                prob[c] = itk.GetArrayFromImage(itkProb)
        else:
            prob = run_output.copy()
        pmin = prob.min()
        pmax = prob.max()
        prange = pmax - pmin
        prob = (prob - pmin) / prange
        for c in range(1,self.num_classes):
            prob[c] = prob[c] * self.class_prior[c]
            class_array = np.argmax(prob, axis=0)
            k = self.class_morph[c]*2
            if k < 2:
                k = 2
            class_array[:k,:] = 0
            class_array[:,:k] = 0
            class_array[-k:,:] = 0
            class_array[:,-k:] = 0
            count = np.count_nonzero(class_array == c)
            if self.class_max_size[c] > 0:
                done = False
                op_iter = 0
                op_iter_max = 40
                while not done and op_iter < op_iter_max:
                    done = True
                    while count < self.class_min_size[c] and op_iter < op_iter_max:
                        prob[c] = prob[c] * 1.05
                        class_array = np.argmax(prob, axis=0)
                        class_array[:k,:] = 0
                        class_array[:,:k] = 0
                        class_array[-k:,:] = 0
                        class_array[:,-k:] = 0
                        count = np.count_nonzero(class_array == c)
                        op_iter += 1
                        done = False
                    while count > self.class_max_size[c] and op_iter < op_iter_max:
                        prob[c] = prob[c] * 0.95
                        class_array = np.argmax(prob, axis=0)
                        class_array[:k,:] = 0
                        class_array[:,:k] = 0
                        class_array[-k:,:] = 0
                        class_array[:,-k:] = 0
                        count = np.count_nonzero(class_array == c)
                        op_iter += 1
                        done = False
        #denom = np.sum(prob, axis=0)
        #denom = np.where(denom == 0, 1, denom)
        #prob =  prob / denom

        return prob

    def classify_probabilities_array(self, prob):
        class_array = np.argmax(prob, axis=0)
        k = max(self.class_morph)*2
        if k < 2:
            k = 2
        class_array[:k,:] = 0
        class_array[:,:k] = 0
        class_array[-k:,:] = 0
        class_array[:,-k:] = 0
        class_image = itk.GetImageFromArray(
            class_array.astype(np.short))

        #itk.imwrite(
            #itk.GetImageFromArray(prob_total.astype(np.float32)),
            #"prob_total_f" + str(self.vfold_num) +
            #"i" + str(image_num) + ".mha")
        #itk.imwrite(class_image,
            #"class_image_init_f" + str(self.vfold_num) +
            #"i" + str(image_num) + ".mha")

        for c in range(1,self.num_classes):
            self.ImageMathS2.SetInput(class_image)
            self.ImageMathS2.Threshold(c, c, 1, 0)
            if self.class_morph[c] > 0:
                self.ImageMathS2.Dilate(self.class_morph[c], 1, 0)
                self.ImageMathS2.Erode(self.class_morph[c], 1, 0)
            class_clean_image = self.ImageMathS2.GetOutputShort()

            if self.class_keep_only_largest[c]==1:
                seg = itk.itkARGUS.SegmentConnectedComponents.New(
                    Input=class_clean_image
                )
                seg.SetKeepOnlyLargestComponent(True)
                seg.Update()
                class_clean_image = seg.GetOutput()

            class_clean_array = itk.GetArrayFromImage(class_clean_image)
            class_array_fill = np.nonzero(class_clean_array)
            
            class_array = np.where(class_array == c, 0, class_array)
            class_array[class_array_fill] = c

        return class_array.astype(np.short)
    
    def inference(self):
        roi_size = (self.size_x, self.size_y)
        prob_size = (self.num_classes, self.size_x, self.size_y)
        prob_total = np.zeros(prob_size)
        with torch.no_grad():
            for m in range(self.num_models):
                test_outputs = sliding_window_inference(
                    self.input_tensor[0].to(self.device), roi_size, 1, self.model[m])
                prob = self.clean_probabilities_array(test_outputs[0].cpu())
                prob_total += prob
        prob_total /= self.num_models
        self.prob_array = self.clean_probabilities_array(prob_total, use_blur=False)
        
        self.class_array = self.classify_probabilities_array(self.prob_array)
        
        return self.class_array
