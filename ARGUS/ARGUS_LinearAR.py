#!/usr/bin/env python
# coding: utf-8

from time import perf_counter
from contextlib import contextmanager

import itk

from ARGUSUtils import *
from ARGUSUtils_IO import *
from ARGUSUtils_Linearization import *
from ARGUSUtils_ARNet import *
from ARGUSUtils_ROINet import *


filename = "../Data/Final15/BAMC-PTXNoSliding/Do_Not_Use/219ns_image_1895283541879_clean.mov"
#filename = "../Data/Final15/BAMC-PTXSliding/212s_image_128692595484031_CLEAN.mov"


height,width = shape_video(filename)

with time_this("Load Video:"):
    us_video = load_video(filename,height,width)
with time_this("Linearization Video:"):
    us_video_linear = linearize_video(us_video).transpose([2,1,0])


device = torch.device("cpu")


arnet_model_filename = "./Models/BAMC_PTX_ARUNET-3D-PR-Final15/best_model.vfold_0.pth"

with time_this("Load ARNet model"):
    arnet_model = arnet_load_model(arnet_model_filename, device)


with time_this("Preprocess for ARNet"):
    arnet_input_tensor = arnet_preprocess_video(us_video_linear)

itk.imwrite(itk.GetImageFromArray(arnet_input_tensor[0,0,:,:,:]),
            "ARUNet_preprocessed_input.mha")


with time_this("ARNet Inference Time:"):
    arnet_output = arnet_inference(arnet_input_tensor, arnet_model, device)
    
itk.imwrite(itk.GetImageFromArray(arnet_output), "ARUNet_output.mha")


with time_this("CPU ROI Extraction Time:"):
    roinet_input_roi = roinet_segment_roi(arnet_input_tensor, arnet_output)

itk.imwrite(itk.GetImageFromArray(roinet_input_roi),
            "ROINet_input_roi.mha")


roinet_model_filename = "./Models/BAMC_PTX_ROINet-StdDevExtended-ExtrudedNS-Final15/best_model.vfold_0.pth"

with time_this("Load ROINet model"):
    roinet_model = roinet_load_model(roinet_model_filename, device)


with time_this("Preprocess for ROINet"):
    roinet_input_tensor = roinet_preprocess_roi(roinet_input_roi)

itk.imwrite(itk.GetImageFromArray(roinet_input_tensor[0,:,:,:]),
            "ROINet_preprocessed_input.mha")


with time_this("ROINet Inference Time:"):
    decision, not_sliding_count, sliding_count, class_array = roinet_inference(
        roinet_input_tensor, roinet_model, device, True)

itk.imwrite( itk.GetImageFromArray(class_array), "ARGUS_output.mha")
print(decision, not_sliding_count, sliding_count)
