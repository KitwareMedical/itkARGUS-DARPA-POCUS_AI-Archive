from doctest import run_docstring_examples
import warnings 
warnings.filterwarnings("ignore")

from monai.utils import first, set_determinism
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Invertd,
    LabelFilterd,
    Lambdad,
    LoadImaged,
    RandFlipd,
    RandSpatialCropd,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
    SpatialCrop,
    SpatialCropd,
    ToTensord,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
from glob import glob

import numpy as np
import cv2

import itk

import sys

import argparse
import site
import ubelt as ub
import pint
site.addsitedir('../../../ARGUS')
from ARGUS_Transforms import ARGUS_RandSpatialCropSlicesd  # NOQA
Ureg = pint.UnitRegistry()


def main():
    parser = argparse.ArgumentParser(description="PNB Experiment", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpuid",type=int, default=0,help="GPU Device ID")
    parser.add_argument("--runid", type=int, default=0, help="Experiment Run ID")
    args = vars(parser.parse_args())
    
    device_num = args["gpuid"]
    run_id = args["runid"]
    print("GPU ID: " + str(device_num))
    num_devices = 1


    img1_dir = "../../Data_ONSD/Preprocessed/"
    img2_dir = "../../Data_ONSD/Annotations/"
        
    all_images = sorted(glob(os.path.join(img1_dir, '*_cropM.mha')))
    all_labels = sorted(glob(os.path.join(img2_dir, '*.overlay.mha')))

    num_folds = 8

    num_classes = 3

    max_epochs = 1500

    net_dims = 2 # Spatial Dimensions
    net_in_channels = 3 # Mean, Std, RawFrame
    net_channels=(16, 32, 64, 128, 32)
    net_strides=(2, 2, 2, 2)

    num_workers_tr = 4
    batch_size_tr = 8
    num_workers_vl = 2
    batch_size_vl = 2

    num_slice = 32
    size_x = 320
    size_y = 640


    model_filename_base = "./results/BAMC_PNB-CLEANED_ARUNet-3D-Mean_StdDevOverTime_RawFrame-yuriArteryData-" + str(num_slice) + "s-VFold-Run" + str(run_id)

    if not os.path.exists(model_filename_base):
        os.makedirs(model_filename_base)
    model_filename_base = model_filename_base+"/"

    num_images = len(all_images)
    print("Num images / labels =", num_images, len(all_labels))

    p_prefix = [
        ' 158',
        ' 64',
        ' 139',
        ' 87',
        ' 117',
        ' 130',
        ' 205',
        ' 209',
        ' 213'
    ]
    n_prefix = [
        ' 23',
        ' 108',
        ' 119',
        ' 129',
        ' 163',
        ' 191',
        ' 228',
        ' 250'
    ]

    fold_prefix_list = []
    p_count = 0
    n_count = 0
    for i in range(num_folds):
        if i%2 == 0:
            num_p = 1
            num_n = 1
            if i > num_folds-2:
                num_p = 2
        else:
            num_p = 1
            num_n = 1
        f = []
        if p_count < len(p_prefix):
            for p in range(num_p):
                f.append([p_prefix[p_count+p]])
        p_count += num_p
        if n_count < len(n_prefix):
            for n in range(num_n):
                f.append([n_prefix[n_count+n]])
        n_count += num_n
        fold_prefix_list.append(f)

    for i in range(num_folds):
        print(i, fold_prefix_list[i])

    total_bytes = 0
    for p in all_images:
        p = ub.Path(p)
        total_bytes += p.stat().st_size
    print((total_bytes * Ureg.byte).to('GiB'))

    total_bytes = 0
    for p in all_labels:
        p = ub.Path(p)
        total_bytes += p.stat().st_size
    print((total_bytes * Ureg.byte).to('GiB'))

    train_files = []
    val_files = []
    test_files = []

    for i in range(num_folds):
        tr_folds = []
        va_folds = []
        for f in range(i,i+num_folds-2):
            tr_folds.append(fold_prefix_list[f%num_folds])
        tr_folds = list(np.concatenate(tr_folds).flat)
        for f in range(i+num_folds-2, i+num_folds-1):
            va_folds.append(fold_prefix_list[f%num_folds])
        va_folds = list(np.concatenate(va_folds).flat)
        te_folds = list(np.concatenate(fold_prefix_list[(i+num_folds-1) % num_folds]).flat)
        train_files.append(
                [
                    {"image": img, "label": seg}
                    for img, seg in zip(
                        [im for im in all_images if any(pref in im for pref in tr_folds)],
                        [se for se in all_labels if any(pref in se for pref in tr_folds)])
                ]
            )
        val_files.append(
                [
                    {"image": img, "label": seg}
                    for img, seg in zip(
                        [im for im in all_images if any(pref in im for pref in va_folds)],
                        [se for se in all_labels if any(pref in se for pref in va_folds)])
                ]
            )
        test_files.append(
                [
                    {"image": img, "label": seg}
                    for img, seg in zip(
                        [im for im in all_images if any(pref in im for pref in te_folds)],
                        [se for se in all_labels if any(pref in se for pref in te_folds)])
                ]
            )
        print(len(train_files[i]),len(val_files[i]),len(test_files[i]))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys='image'),
            AsChannelFirstd(keys='label'),
            ScaleIntensityRanged(
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                keys=["image"]),
            ARGUS_RandSpatialCropSlicesd(
                num_slices=[num_slice,1],
                axis=0,
                reduce_to_statistics=[True,False],
                extended=False,
                include_center_slice=True,
                include_mean_abs_diff=False,
                keys=['image','label']),
            Resized(
                spatial_size=(-1,size_y),
                mode=["bilinear","nearest"],
                keys=['image','label']
            ),
            RandSpatialCropd(
                roi_size=(size_x,size_y),
                random_center=True,
                random_size=False,
                keys=['image','label']
            ),
            RandFlipd(prob=0.5, 
                spatial_axis=0,
                keys=['image', 'label']),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys='image'),
            AsChannelFirstd(keys='label'),
            ScaleIntensityRanged(
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                keys=["image"]),
            ARGUS_RandSpatialCropSlicesd(
                num_slices=[num_slice,1],
                center_slice=-num_slice/2 - 1,
                axis=0,
                reduce_to_statistics=[True,False],
                extended=False,
                include_center_slice=True,
                include_mean_abs_diff=False,
                keys=['image','label']),
            Resized(
                spatial_size=(-1,size_y),
                mode=["bilinear","nearest"],
                keys=['image','label']
            )
        ]
    )

    device = torch.device("cuda:"+str(device_num))

    for i in range(0,num_folds,num_devices):
        train_ds = CacheDataset(data=train_files[i], transform=train_transforms,cache_rate=1.0, num_workers=num_workers_tr)
        train_loader = DataLoader(train_ds, batch_size=batch_size_tr, shuffle=True, num_workers=num_workers_tr)
        val_ds = CacheDataset(data=val_files[i], transform=val_transforms, cache_rate=1.0, num_workers=num_workers_vl)
        val_loader = DataLoader(val_ds, batch_size=batch_size_vl, num_workers=num_workers_vl)
        vfold_train(i, train_loader, val_loader,net_dims, net_in_channels,num_classes, net_channels, net_strides, device, max_epochs, train_ds, size_x, size_y, num_slice, batch_size_vl, model_filename_base)
        train_loader.dataset._cache.clear()
        val_loader.dataset._cache.clear()

def vfold_train(vfold_num, train_loader, val_loader, net_dims, net_in_channels,num_classes, net_channels, net_strides, device, max_epochs, train_ds, size_x, size_y, num_slices, batch_size_vl, model_filename_base):
    model = UNet(
        dimensions=net_dims,
        in_channels=net_in_channels,
        out_channels=num_classes,
        channels=net_channels,
        strides=net_strides,
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=num_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=num_classes)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"{vfold_num}: epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"],
                batch_data["label"]
            )
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"{vfold_num} epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"],
                        val_data["label"]
                    )
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    roi_size = (size_x, size_y)
                    sw_batch_size = batch_size_vl
                    val_outputs = sliding_window_inference(
                       val_inputs, roi_size, sw_batch_size, model)
                    # val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if epoch > 100:
                    metric = (metric_values[-1]+metric_values[-2]+metric_values[-3])/3
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), model_filename_base+'best_model.vfold_'+str(vfold_num)+'.pth')
                        print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
                torch.save(model.state_dict(), model_filename_base+'last_model.vfold_'+str(vfold_num)+'.pth')

    np.save(model_filename_base+"loss_"+str(vfold_num)+".npy", epoch_loss_values)
    np.save(model_filename_base+"val_dice_"+str(vfold_num)+".npy", metric_values)

if __name__ == "__main__":
    main()