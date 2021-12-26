import numpy as np

import itk
from itk import TubeTK as ttk

import torch

from monai.networks.nets import UNet
from monai.transforms import ( ScaleIntensityRange, ToTensor )
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

from ARGUSUtils_Transforms import *


def roinet_segment_roi(arnet_input_tensor, arnet_output):
    roi_size_x = 160

    roi_min_x = 0
    roi_max_x = arnet_output.shape[0]-1

    while( np.count_nonzero(arnet_output[roi_min_x,:,:]==1)==0
           and roi_min_x<roi_max_x ):
        roi_min_x += 1
    while( np.count_nonzero(arnet_output[roi_max_x,:,:]==1)==0
           and roi_max_x>roi_min_x):
        roi_max_x -= 1
    roi_mid_x = (roi_min_x + roi_max_x)//2
    roi_min_x = max(roi_mid_x-roi_size_x//2,0)
    roi_max_x = min(roi_min_x+roi_size_x,arnet_output.shape[0]-1)
    roi_min_x = roi_max_x-roi_size_x

    roi_array = arnet_input_tensor[0,0,roi_min_x:roi_max_x,:,:].numpy()
    roi_array = roi_array.transpose([2,0,1])
    return roi_array

def roinet_load_model(filename, device):
    num_classes = 3

    net_in_dims = 2
    net_in_channels = 4
    net_channels=(32, 64, 128)
    net_strides=(2, 2)

    model = UNet(
        dimensions=net_in_dims,
        in_channels=net_in_channels,
        out_channels=num_classes,
        channels=net_channels,
        strides=net_strides,
        num_res_units=2,
        norm=Norm.BATCH,
        ).to(device)    
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()

    return model

def roinet_preprocess_roi(roi_array):
    num_slices = 32

    net_in_channels = 4

    roi_input_array = np.empty([1, net_in_channels,
                                roi_array.shape[1], roi_array.shape[2]])

    Scale = ScaleIntensityRange(
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0)
    roi_array_scaled = Scale(roi_array)

    Crop = ARGUS_RandSpatialCropSlices(
        num_slices=num_slices,
        axis=0,
        reduce_to_statistics=True,
        extended=True)
    roi_input_array[0] = Crop(roi_array_scaled)

    ConvertToTensor = ToTensor()
    roi_input_tensor = ConvertToTensor(roi_input_array.astype(np.float32))

    return roi_input_tensor

def roinet_inference(roinet_input_tensor, roinet_model, device, debug):
    num_classes = 3

    class_not_sliding = 1
    class_sliding = 2

    class_prior = [1,1,1] #[1.3,1.0,0.85]

    num_slices = 32

    size_x = 160
    size_y = 320
    roi_size = (size_x, size_y)

    min_size = 1100
    max_size = 1600

    with torch.no_grad():
        test_outputs = sliding_window_inference(
            roinet_input_tensor.to(device), roi_size, 1, roinet_model)
        prob_shape = test_outputs[0,:,:,:].shape
        prob = np.empty(prob_shape)
        prob = test_outputs[0,:,:,:].numpy()

        class_array = np.zeros(prob[0].shape)
        pmin = prob[0].min()
        pmax = prob[0].max()
        for c in range(1,num_classes):
            pmin = min(pmin, prob[c].min())
            pmax = max(pmax, prob[c].max())
        prange = pmax - pmin
        prob = (prob - pmin) / prange

        for c in range(num_classes):
            prob[c] = prob[c] * class_prior[c]
        done = False
        while not done:
            done = True
            count = np.count_nonzero(class_array>0)
            print(count)
            while count<min_size:
                prob[class_sliding] = prob[class_sliding] * 1.05
                prob[class_not_sliding] = prob[class_not_sliding] * 1.05
                class_array = np.argmax(prob,axis=0)
                count = np.count_nonzero(class_array>0)
                done = False
                print(count)
            while count>max_size:
                prob[class_sliding] = prob[class_sliding] * 0.95
                prob[class_not_sliding] = prob[class_not_sliding] * 0.95
                class_array = np.argmax(prob,axis=0)
                count = np.count_nonzero(class_array>0)
                done = False
                print(count)

        class_array = np.argmax(prob,axis=0).astype(np.float32)
        
        sliding_count = np.count_nonzero(class_array==class_sliding)
        not_sliding_count = np.count_nonzero(class_array==class_not_sliding)
        if( not_sliding_count > sliding_count ):
            return "Not Sliding", not_sliding_count, sliding_count, class_array
        else:
            return "Sliding", not_sliding_count, sliding_count, class_array
