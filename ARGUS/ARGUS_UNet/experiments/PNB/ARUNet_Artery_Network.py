import warnings
warnings.filterwarnings("ignore")

import torch
from monai.utils import first, set_determinism
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadImaged,
    RandFlipd,
    RandSpatialCropd,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)
import numpy as np
import os
from glob import glob
import site
site.addsitedir('/data/barry.ravichandran/repos/AnatomicRecon-POCUS-AI/ARGUS')
from ARGUS_Transforms import ARGUS_RandSpatialCropSlicesd  # NOQA

class ARUNet_Artery_Network:
    
    def __init__(self):
        self.filename_base = "ARUNet-Artery-VFold-Training"
        self.img_file_extension = "*_cropM.nii.gz"
        self.label_file_extension = "*.overlay.mha"

        self.num_classes = 2

        self.max_epochs = 1500

        self.num_folds = 10

        self.net_dims = 2

        # Mean, Std, RawFrame
        self.net_in_channels = 3

        self.net_channels=(16, 32, 64, 128, 32)
        self.net_strides=(2, 2, 2, 2)

        self.cache_rate_train = 1.0
        self.num_workers_train = 4
        self.batch_size_train = 8
        
        self.cache_rate_val = 1.0
        self.num_workers_val = 2
        self.batch_size_val = 2

        self.cache_rate_test = 1.0
        self.num_workers_test = 2
        self.batch_size_test = 2

        self.num_slices = 16

        self.size_x = 320
        self.size_y = 640

        self.all_images = []
        self.all_labels = []

        self.class_artery = 1

        self.class_min_size = np.zeros(self.num_classes)
        self.class_max_size = np.zeros(self.num_classes)
        self.class_min_size[self.class_artery] = 20000
        self.class_max_size[self.class_artery] = 30000
        
        self.erosion_size = 5
        self.dilation_size = 5

        self.train_transforms = Compose(
            [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys='image'),
            AsChannelFirstd(keys='label'),
            ScaleIntensityRanged(
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                keys=["image"]),
            ARGUS_RandSpatialCropSlicesd(
                num_slices=[self.num_slices,1],
                axis=0,
                reduce_to_statistics=[True,False],
                require_labeled=True,
                extended=False,
                include_center_slice=True,
                keys=['image','label']),
            Resized(
                spatial_size=(-1,self.size_y),
                mode=["bilinear","nearest"],
                keys=['image','label']),
            RandSpatialCropd(
                roi_size=(self.size_x,self.size_y),
                random_center=True,
                random_size=False,
                keys=['image','label']),
            RandFlipd(prob=0.5, 
                spatial_axis=0,
                keys=['image', 'label']),
            ToTensord(keys=["image", "label"], dtype=torch.float)
            ])
        
        self.val_transforms = Compose(
            [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys='image'),
            AsChannelFirstd(keys='label'),
            ScaleIntensityRanged(
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                keys=["image"]),
            ARGUS_RandSpatialCropSlicesd(
                num_slices=[self.num_slices,1],
                center_slice=-self.num_slices/2 - 1,
                axis=0,
                reduce_to_statistics=[True,False],
                extended=False,
                include_center_slice=True,
                keys=['image','label']),
            Resized(
                spatial_size=(-1,self.size_y),
                mode=["bilinear","nearest"],
                keys=['image','label']),
            ToTensord(keys=["image", "label"],dtype=torch.float)
            ])

        self.test_transforms = Compose(
            [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys='image'),
            AsChannelFirstd(keys='label'),
            ScaleIntensityRanged(
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                keys=["image"]),
            ARGUS_RandSpatialCropSlicesd(
                num_slices=[self.num_slices,1],
                center_slice=-self.num_slices/2 - 1,
                axis=0,
                reduce_to_statistics=[True,False],
                extended=False,
                include_center_slice=True,
                keys=['image','label']),
            Resized(
                spatial_size=(-1,self.size_y),
                mode=["bilinear","nearest"],
                keys=['image','label']),
            ToTensord(keys=["image", "label"],dtype=torch.float)
            ])