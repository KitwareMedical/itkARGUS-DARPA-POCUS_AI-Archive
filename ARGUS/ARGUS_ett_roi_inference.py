import itk
from itk import TubeTK as tube

import numpy as np

import torch

from ARGUS_classification_inference import ARGUS_classification_inference

from ARGUS_preprocess_butterfly import ARGUS_preprocess_butterfly
from ARGUS_preprocess_sonosite import ARGUS_preprocess_sonosite
from ARGUS_preprocess_clarius import ARGUS_preprocess_clarius

class ARGUS_ett_roi_inference(ARGUS_classification_inference):
    def __init__(self, config_file_name="ARGUS_taskid.cfg", network_name="final", device_num=0, source=None):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_ett_video = []
        if source=="Butterfly" or source==None:
            self.preprocess_ett = ARGUS_preprocess_butterfly(new_size=[self.size_x, self.size_y])
        elif source=="Sonosite":
            self.preprocess_ett = ARGUS_preprocess_sonosite(new_size=[self.size_x, self.size_y])
        elif source=="Clarius":
            self.preprocess_ett = ARGUS_preprocess_clarius(new_size=[self.size_x, self.size_y])

        self.number_of_seconds = 10.0
        self.minimum_number_of_positive_seconds = 3.0
        
    def preprocess(self, vid, lbl=None, slice_num=None, crop_data=True, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_ett_video = self.preprocess_ett.process( vid )
        else:
            self.preprocessed_ett_video = vid
        super().preprocess(self.preprocessed_ett_video, lbl, slice_num, scale_data, rotate_data)

    def volume_preprocess(self, vid, crop_data=True, slice_num=None, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_ett_video = self.preprocess_ett.process( vid )
        else:
            self.preprocessed_ett_video = vid
            
        self.ARGUS_Preprocess._gradient_cache = None
        self.ARGUS_Preprocess.cache_gradient = True
        
        vid_img = self.preprocessed_ett_video
        
        ImageF = itk.Image[itk.F, 3]
        ImageSS = itk.Image[itk.SS, 3]

        img_size = vid_img.GetLargestPossibleRegion().GetSize()

        resample = tube.ResampleImage[ImageF].New()
        resample.SetInput(vid_img)
        size = [self.size_x, self.size_y, img_size[2]]
        resample.SetSize(size)
        resample.Update()
        vid_roi_img = resample.GetOutput()

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

        self.input_image = vid_roi_img

    def volume_inference(self, step=10, slice_min=None, slice_max=None, use_cache=True):
        img_size = self.input_image.GetLargestPossibleRegion().GetSize()
        img_spacing = self.input_image.GetSpacing()

        self.prob_array  = []
        self.classification_array = []
        self.prob_total  = np.zeros(self.num_classes)
        if not use_cache:
            self.ARGUS_Preprocess._gradient_cache = None
            self.ARGUS_Preprocess.cache_gradient = True
            
        num_slices = 0
        window_size = self.num_slices
        resize_window = False
        if img_size[2] < window_size-1:
            print("Short video: reducing window size")
            window_size = img_size[2]-1
            self.ARGUS_Preprocess.num_slices = window_size
            resize_window = True
        window_buffer = window_size // 2 + 1
        if slice_min == None:
            slice_min = int(img_size[2] - self.number_of_seconds/img_spacing[2])
        if slice_min < window_buffer:
            print("Short video: reducing search size min")
            slice_min = window_buffer
        if slice_max == None:
            slice_max = img_size[2] - window_buffer
        if slice_max < slice_min:
            print("Short video: reducing search size max")
            slice_max = slice_min+1
        for slice_num in range(slice_min, slice_max, step):
            vid_roi_array = itk.GetArrayFromImage(self.input_image)
            self.ARGUS_Preprocess.center_slice = slice_num
            roi_array = self.ARGUS_Preprocess(vid_roi_array)

            ar_input_array = np.empty([1,
                                       1,
                                       self.net_in_channels,
                                       self.size_x,
                                       self.size_y])
            ar_input_array[0,0] = roi_array
    
            self.input_tensor = self.ConvertToTensor(ar_input_array.astype(np.float32))

            prob_total = np.zeros(self.num_classes)
            with torch.no_grad():
                for m in range(self.num_models):
                    test_outputs = self.model[m](self.input_tensor[0].to(self.device))
                    prob = self.clean_probabilities(test_outputs[0].cpu().detach().numpy())
                    prob_total += prob
            prob_total /= self.num_models
            self.prob_array.append(self.clean_probabilities(prob_total))
            self.classification_array.append(self.classify_probabilities(prob_total))
            self.prob_total += prob_total
            num_slices += 1

        if resize_window:
            self.ARGUS_Preprocess.num_slices = self.num_slices
            
        self.prob_total /= num_slices

        frames = len(self.classification_array)
        frames_positive = np.count_nonzero(self.classification_array)
        frames_negative = frames-frames_positive

        frames_positive_threshold = (self.minimum_number_of_positive_seconds/self.number_of_seconds) * frames

        self.classification = 0
        if frames_positive > frames_positive_threshold:
            self.classification = 1

        return self.classification, [frames_negative, frames_positive]
