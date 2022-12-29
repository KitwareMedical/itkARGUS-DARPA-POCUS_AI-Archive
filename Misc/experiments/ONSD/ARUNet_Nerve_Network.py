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

class ARUNet_Nerve_Network:
    
    def __init__(self):
        self.target_data = "ONSD"
        self.filename_base = "ARUNet-ONSD-Nerve-VFold-Training-320x480-32Videos"
        self.img_file_extension = "*_cropM.nii.gz"
        self.label_file_extension = "*.overlay.nii.gz"
        self.result_files_savepath = "/data/barry.ravichandran/repos/AnatomicRecon-POCUS-AI/ARGUS/ARGUS_UNet/Results/ARUNet-ONSD-Nerve-VFold-Training-320x480-32Videos"

        self.num_classes = 3

        self.max_epochs = 1500

        self.num_folds = 15

        self.net_dims = 2

        # Mean, Std, RawFrame, Gradient
        self.net_in_channels = 6

        self.net_channels=(16, 32, 64, 128, 32)
        self.net_strides=(2, 2, 2, 2)

        self.cache_rate_train = 1.0
        self.num_workers_train = 8
        self.batch_size_train = 4
        
        self.cache_rate_val = 1.0
        self.num_workers_val = 2
        self.batch_size_val = 2

        self.cache_rate_test = 1.0
        self.num_workers_test = 2
        self.batch_size_test = 2

        self.num_slices = 3

        self.size_x = 320
        self.size_y = 480

        self.all_images = []
        self.all_labels = []

        self.class_nerve = 2

        self.class_min_size = np.zeros(self.num_classes)
        self.class_max_size = np.zeros(self.num_classes)
        self.class_min_size[self.class_nerve] = 0
        self.class_max_size[self.class_nerve] = 7000
        
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
                include_gradient=True,
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
                include_gradient=True,
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
                include_gradient=True,
                keys=['image','label']),
            ToTensord(keys=["image", "label"],dtype=torch.float)
            ])