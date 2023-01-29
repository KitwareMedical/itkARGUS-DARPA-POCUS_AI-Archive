import itk
from itk import TubeTK as tube

import numpy as np

import torch

from monai.inferers import sliding_window_inference

from ARGUS_segmentation_inference import ARGUS_segmentation_inference

from ARGUS_preprocess_butterfly import ARGUS_preprocess_butterfly
from ARGUS_preprocess_sonosite import ARGUS_preprocess_sonosite
from ARGUS_preprocess_clarius import ARGUS_preprocess_clarius

class ARGUS_onsd_ar_inference(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_onsd_ar.cfg", network_name="final", device_num=0, source=None):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_onsd_video = []
        if source=="Butterfly" or source==None:
            self.preprocess_onsd = ARGUS_preprocess_butterfly(new_size=[self.size_x, self.size_y])
        elif source=="Sonosite":
            self.preprocess_onsd = ARGUS_preprocess_sonosite(new_size=[self.size_x, self.size_y])
        elif source=="Clarius":
            self.preprocess_onsd = ARGUS_preprocess_clarius(new_size=[self.size_x, self.size_y])
        
    def preprocess(self, vid, lbl=None, slice_num=None, crop_data=True, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_onsd_video = self.preprocess_onsd.process( vid )
        else:
            self.preprocessed_onsd_video = vid
            
        super().preprocess(self.preprocessed_onsd_video, lbl, slice_num, scale_data, rotate_data)

    def volume_preprocess(self, vid, lbl_img=None, crop_data=True, slice_num=None, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_onsd_video = self.preprocess_onsd.process( vid )
        else:
            self.preprocessed_onsd_video = vid
            
        self.ARGUS_Preprocess._gradient_cache = None
        self.ARGUS_Preprocess.cache_gradient = True
        
        vid_img = self.preprocessed_onsd_video
        
        ImageF = itk.Image[itk.F, 3]
        ImageSS = itk.Image[itk.SS, 3]

        img_size = vid_img.GetLargestPossibleRegion().GetSize()

        resample = tube.ResampleImage[ImageF].New()
        resample.SetInput(vid_img)
        size = [self.size_x, self.size_y, img_size[2]]
        resample.SetSize(size)
        resample.Update()
        vid_roi_img = resample.GetOutput()
        if lbl_img != None:
            resample = tube.ResampleImage[ImageSS].New()
            resample.SetInput(lbl_img)
            size = [self.size_x, self.size_y, img_size[2]]
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

        self.input_image = vid_roi_img
        if lbl_img != None:
            self.label_image = lbl_roi_img
        else:
            self.label_image = None

    def volume_inference(self, step=5, slice_min=None, slice_max=None, use_cache=False):
        roi_size = (self.size_x, self.size_y)

        img_size = self.input_image.GetLargestPossibleRegion().GetSize()
        img_shape = self.input_image.shape

        prob_size = [self.num_classes, img_shape[0], img_size[1], img_shape[2]]

        if not use_cache:
            self.prob_array  = np.zeros(prob_size)
            self.class_array = np.zeros(self.input_image.shape)
            self.ARGUS_Preprocess._gradient_cache = None
            self.ARGUS_Preprocess.cache_gradient = True
            
        if slice_min == None:
            slice_min = self.num_slices//2+1
        if slice_max == None:
            slice_max = img_size[2] - self.num_slices//2-1
        for slice_num in range(slice_min, slice_max, step):
            vid_roi_array = itk.GetArrayFromImage(self.input_image)
            self.ARGUS_Preprocess.center_slice = slice_num
            roi_array = self.ARGUS_Preprocess(vid_roi_array)

            if self.label_image != None:
                lbl_array = itk.GetArrayFromImage(self.label_image)[slice_num]
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
    
            self.input_tensor = self.ConvertToTensor(ar_input_array.astype(np.float32))
            self.label_tensor = self.ConvertToTensor(ar_lbl_array.astype(np.short))

            prob_size = (self.num_classes, self.size_x, self.size_y)
            prob_total = np.zeros(prob_size)
            with torch.no_grad():
                for m in range(self.num_models):
                    test_outputs = sliding_window_inference(
                        self.input_tensor[0].to(self.device), roi_size, 1, self.model[m])
                    prob = self.clean_probabilities_array(test_outputs[0].cpu())
                    prob_total += prob
            prob_total /= self.num_models
            tmp_prob_array = self.clean_probabilities_array(prob_total, use_blur=False)
            for c in range(self.num_classes):
                self.prob_array[c][slice_num] = tmp_prob_array[c]
    
            self.class_array[slice_num-step//2:slice_num+step//2+1] = self.classify_probabilities_array(tmp_prob_array)

        return self.class_array
