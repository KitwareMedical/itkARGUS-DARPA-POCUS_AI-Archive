import warnings
warnings.filterwarnings("ignore")

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.utils import first, set_determinism
from monai.transforms import (
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    EnsureType,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
    SpatialResampled,
    SpatialCrop,
    SpatialCropd,
    ToTensord,
)
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import PersistentDataset, CacheDataset, DataLoader, Dataset, decollate_batch, list_data_collate

import torch

import configparser

import os
import json
import numpy as np

from glob import glob

import random

import pathlib

import itk

import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from ARGUS_segmentation_inference import ARGUS_segmentation_inference
from ARGUS_Transforms import ARGUS_RandSpatialCropSlicesd

class ARGUS_segmentation_train(ARGUS_segmentation_inference):
    def __init__(self, config_file_name, network_name="vfold", device_num=0):
        
        super().__init__(config_file_name, network_name, device_num)
        
        config = configparser.ConfigParser()
        config.read(config_file_name)

        self.image_dirname = json.loads(config[network_name]['image_dirname'])
        self.image_filesuffix = config[network_name]['image_filesuffix']
        self.label_dirname = json.loads(config[network_name]['label_dirname'])
        self.label_filesuffix = config[network_name]['label_filesuffix']
        
        self.train_data_portion = float(config[network_name]['train_data_portion'])
        self.validation_data_portion = float(config[network_name]['validation_data_portion'])
        self.test_data_portion = float(config[network_name]['test_data_portion'])
        
        self.num_folds = int(config[network_name]['num_folds'])
        tmp_str = config[network_name]['randomize_folds']
        self.randomize_folds = False
        if tmp_str == "True":
            self.randomize_folds = True
        if self.num_folds < 4:
            self.num_folds = 1
        self.refold_interval = int(config[network_name]['refold_interval'])
        
        self.validation_interval = int(config[network_name]['validation_interval'])
        
        self.pos_prefix = json.loads(config[network_name]['pos_prefix'])
        self.neg_prefix = json.loads(config[network_name]['neg_prefix'])

        self.results_filename_base = config[network_name]['results_filename_base']
        self.results_dirname = config[network_name]['results_dirname']
        
        tmp_str = config[network_name]['use_persistent_cache']
        self.use_persistent_cache = False
        if tmp_str == "True":
            self.use_persistent_cache = True
        
        self.max_epochs = int(config[network_name]['max_epochs'])

        self.cache_rate_train = 1
        self.num_workers_train = 6
        self.batch_size_train = 12

        self.cache_rate_val = 1
        self.num_workers_val = 4
        self.batch_size_val = 2

        self.cache_rate_test = 0
        self.num_workers_test = 1
        self.batch_size_test = 1

        self.all_train_images = []
        self.all_train_labels = []

        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AsChannelFirstd(keys=["image","label"]),
                Resized(
                    spatial_size=[self.size_y,self.size_x],
                    mode=['bilinear','nearest-exact'],
                    keys=["image", "label"],
                ),
                RandRotated(prob=0.2,
                    range_z=0.15,
                    keep_size=True,
                    keys=['image', 'label'],
                ),
                ARGUS_RandSpatialCropSlicesd(
                    num_slices=[self.num_slices, 1],
                    axis=0,
                    reduce_to_statistics=[self.reduce_to_statistics, False],
                    extended=self.reduce_to_statistics,
                    include_center_slice=self.reduce_to_statistics,
                    include_gradient=self.reduce_to_statistics,
                    keys=["image", "label"],
                ),
                RandFlipd(prob=0.5, spatial_axis=0, keys=["image", "label"]),
                RandZoomd(prob=0.5, 
                    min_zoom=1.0,
                    max_zoom=1.1,
                    keep_size=True,
                    mode=['bilinear', 'nearest-exact'],
                    keys=['image', 'label'],
                ),
                ToTensord(keys=["image", "label"], dtype=torch.float),
            ]
        )

        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AsChannelFirstd(keys=["image","label"]),
                Resized(
                    spatial_size=[self.size_x,self.size_y],
                    mode=['bilinear','nearest-exact'],
                    keys=["image", "label"],
                ),
                ARGUS_RandSpatialCropSlicesd(
                    num_slices=[self.num_slices, 1],
                    center_slice=self.testing_slice,
                    axis=0,
                    reduce_to_statistics=[True, False],
                    extended=True,
                    include_center_slice=True,
                    include_gradient=True,
                    keys=["image", "label"],
                ),
                ToTensord(keys=["image", "label"], dtype=torch.float),
            ]
        )

        self.test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AsChannelFirstd(keys=["image","label"]),
                Resized(
                    spatial_size=[self.size_x,self.size_y],
                    mode=['bilinear','nearest-exact'],
                    keys=["image", "label"],
                ),
                ARGUS_RandSpatialCropSlicesd(
                    num_slices=[self.num_slices, 1],
                    center_slice=self.testing_slice,
                    axis=0,
                    reduce_to_statistics=[True, False],
                    extended=True,
                    include_center_slice=True,
                    include_gradient=True,
                    keys=["image", "label"],
                ),
                ToTensord(keys=["image", "label"], dtype=torch.float),
            ]
        )

    def init_model(self, model_num):
        self.model[model_num] = UNet(
            spatial_dims=self.net_in_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
            channels=self.net_layer_channels,
            strides=self.net_layer_strides,
            num_res_units=self.net_num_residual_units,
            norm=Norm.BATCH,
            ).to(self.device)

    def setup_vfold_files(self):
        self.all_train_images = []
        for dirname in self.image_dirname:
            self.all_train_images = self.all_train_images + sorted(glob(os.path.join(dirname, self.image_filesuffix)))
        self.all_train_labels = []
        for dirname in self.label_dirname:
            self.all_train_labels = self.all_train_labels + sorted(glob(os.path.join(dirname, self.label_filesuffix)))

        num_images = len(self.all_train_images)
        print("Num images / labels =", num_images, len(self.all_train_labels))

        if len(self.pos_prefix) == 0:
            done = False
            basesize = 2
            while not done:
                self.pos_prefix = [os.path.basename(x)[0:basesize] for x in self.all_train_images]
                test_unique = set(self.pos_prefix)
                done = True
                if len(test_unique) != len(self.pos_prefix):
                    basesize += 1
                    done = False
        elif len(self.pos_prefix) < self.num_folds:
            pos_train_images = [x for x in self.all_train_images if any(pref == os.path.basename(x)[0:len(pref)] for pref in self.pos_prefix)]
            done = False
            basesize = 2
            while not done:
                self.pos_prefix = [os.path.basename(x)[0:basesize] for x in pos_train_images]
                test_unique = set(self.pos_prefix)
                done = True
                if len(test_unique) != len(pos_train_images):
                    basesize += 1
                    done = False
        if len(self.neg_prefix) == 0:
            if len(self.pos_prefix) == 0:
                print("ERROR: Cannot resolve training positive and negative instances")
        elif len(self.neg_prefix) < self.num_folds:
            neg_train_images = [x for x in self.all_train_images if any(pref == os.path.basename(x)[:len(pref)] for pref in self.neg_prefix)]
            done = False
            basesize = 2
            while not done:
                self.neg_prefix = [os.path.basename(x)[0:basesize] for x in neg_train_images]
                test_unique = set(self.neg_prefix)
                done = True
                if len(test_unique) != len(neg_train_images):
                    basesize += 1
                    done = False
        num_pos = len(self.pos_prefix)
        num_neg = len(self.neg_prefix)
        print( f"Num pos / Num neg = {num_pos} / {num_neg}" )
        
        if self.randomize_folds==True:
            random.shuffle(self.pos_prefix)
            random.shuffle(self.neg_prefix)
            
        fold_prefix = []
        pos_fold_size = num_pos // self.num_folds
        neg_fold_size = num_neg // self.num_folds
        pos_extra_case = num_pos - (pos_fold_size * self.num_folds)
        neg_extra_case = num_neg - (neg_fold_size * self.num_folds)
        pos_count = 0
        neg_count = 0
        for i in range(self.num_folds):
            pos_fsize = pos_fold_size
            if i<pos_extra_case:
                pos_fsize += 1
            neg_fsize = neg_fold_size
            if i<neg_extra_case:
                neg_fsize += 1
 
            fprefix = []
            for pre in range(pos_fsize):
                fprefix.append(self.pos_prefix[pos_count+pre])
            pos_count += pos_fsize
            for pre in range(neg_fsize):
                fprefix.append(self.neg_prefix[neg_count+pre])
            neg_count += neg_fsize

            fold_prefix.append(fprefix)

        for i in range(self.num_folds):
            print(f"VFold-Prefix[{i}] = {fold_prefix[i]}")

        self.train_files = []
        self.val_files = []
        self.test_files = []

        for i in range(self.num_folds):
            tr_folds = []
            va_folds = []
            te_folds = []
            if self.num_folds == 1:
                if self.train_data_portion == 1.0:
                    tr_folds = fold_prefix[0]
                    te_folds = fold_prefix[0]
                    va_folds = fold_prefix[0]
                else:
                    num_pre = len(fold_prefix[0])
                    num_tr = int(num_pre * self.train_data_portion)
                    num_va = int(num_pre * self.validation_data_portion)
                    num_te = int(num_pre * self.test_data_portion)
                    if self.test_data_portion > 0 and num_te < 1 and num_tr > 2:
                        num_tr -= 1
                        num_te = 1
                    if self.validation_data_portion > 0 and num_va < 1 and num_tr > 2:
                        num_tr -= 1
                        num_va = 1
                    tr_folds = list(fold_prefix[0][0:num_tr])
                    if num_va>0:
                        va_folds = list(fold_prefix[0][num_tr:num_tr+num_va])
                    if num_te>0:
                        te_folds = list(fold_prefix[0][num_tr+num_va:])
            else:
                num_tr = int(self.num_folds * self.train_data_portion)
                num_va = int(self.num_folds * self.validation_data_portion)
                num_te = int(self.num_folds * self.test_data_portion)
                if self.test_data_portion > 0 and num_te < 1 and num_tr > 2:
                    num_tr -= 1
                    num_te = 1
                if self.validation_data_portion > 0 and num_va < 1 and num_tr > 2:
                    num_tr -= 1
                    num_va = 1
                num_tr += self.num_folds - num_tr - num_te - num_va
                
                for f in range(i, i + num_tr):
                    tr_folds.append(fold_prefix[f % self.num_folds])
                tr_folds = list(np.concatenate(tr_folds).flat)
                if num_va > 0:
                    for f in range(i + num_tr, i + num_tr + num_va):
                        va_folds.append(fold_prefix[f % self.num_folds])
                    va_folds = list(np.concatenate(va_folds).flat)
                if num_te > 0:
                    for f in range(i + num_tr + num_va, i + num_tr + num_va + num_te):
                        te_folds.append(fold_prefix[f % self.num_folds])
                    te_folds = list(np.concatenate(te_folds).flat)
            self.train_files.append(
                [
                    {"image": img, "label": seg}
                    for img, seg in zip(
                        [
                            im
                            for im in self.all_train_images
                            if any(pref in im for pref in tr_folds)
                        ],
                        [
                            se
                            for se in self.all_train_labels
                            if any(pref in se for pref in tr_folds)
                        ],
                    )
                ]
            )
            if len(va_folds) > 0:
                self.val_files.append(
                    [
                        {"image": img, "label": seg}
                        for img, seg in zip(
                            [
                                im
                                for im in self.all_train_images
                                if any(pref in im for pref in va_folds)
                            ],
                            [
                                se
                                for se in self.all_train_labels
                                if any(pref in se for pref in va_folds)
                            ],
                        )
                    ]
                )
            if len(te_folds) > 0:
                self.test_files.append(
                    [
                        {"image": img, "label": seg}
                        for img, seg in zip(
                            [
                                im
                                for im in self.all_train_images
                                if any(pref in im for pref in te_folds)
                            ],
                            [
                                se
                                for se in self.all_train_labels
                                if any(pref in se for pref in te_folds)
                            ],
                        )
                    ]
                )
            #print( "**** VFold =", i )
            #print( "   TRAIN", self.train_files[i])
            #print( "   VAL", self.val_files[i])
            #print( "   TEST", self.test_files[i])

    def setup_training_vfold(self, vfold_num, run_num):
        self.vfold_num = vfold_num

        if self.use_persistent_cache:
            persistent_cache = pathlib.Path(
                ".",
                "data_cache",
                self.network_name+"_f"+str(vfold_num)+"_r"+str(run_num)
            )
            persistent_cache.mkdir(parents=True, exist_ok=True)
            train_ds = PersistentDataset(
                data=self.train_files[self.vfold_num],
                transform=self.train_transforms,
                cache_dir=persistent_cache,
            )
        else:
            train_ds = CacheDataset(
                data=self.train_files[self.vfold_num],
                transform=self.train_transforms,
                cache_rate=self.cache_rate_train,
                num_workers=self.num_workers_train,
            )


        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers_train,
            collate_fn=list_data_collate,
            pin_memory=True,
        )

        if len(self.val_files) > self.vfold_num and len(self.val_files[self.vfold_num]) > 0:
            if self.use_persistent_cache:
                val_ds = PersistentDataset(
                    data=self.val_files[self.vfold_num],
                    transform=self.val_transforms,
                    cache_dir=persistent_cache,
                )
            else:
                val_ds = CacheDataset(
                    data=self.val_files[self.vfold_num],
                    transform=self.val_transforms,
                    cache_rate=self.cache_rate_val,
                    num_workers=self.num_workers_val
                )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size_val,
                num_workers=self.num_workers_val,
                collate_fn=list_data_collate,
                pin_memory=True,
            )

    def setup_testing_vfold(self, vfold_num, run_num):
        self.vfold_num = vfold_num

        if len(self.test_files) > self.vfold_num and len(self.test_files[self.vfold_num]) > 0:
            if self.use_persistent_cache:
                persistent_cache = pathlib.Path(
                    ".",
                    "data_cache",
                    self.network_name+"_f"+str(vfold_num)+"_r"+str(run_num)
                    )
                persistent_cache.mkdir(parents=True, exist_ok=True)
                test_ds = PersistentDataset(
                    data=self.test_files[self.vfold_num],
                    transform=self.test_transforms,
                    cache_dir=persistent_cache,
                )
            else:
                test_ds = CacheDataset(
                    data=self.test_files[self.vfold_num],
                    transform=self.test_transforms,
                    cache_rate=self.cache_rate_test,
                    num_workers=self.num_workers_test
                )
            self.test_loader = DataLoader(
                test_ds,
                batch_size=self.batch_size_test,
                num_workers=self.num_workers_test,
                collate_fn=list_data_collate,
                pin_memory=True,
            )

    def train_vfold(self, run_id=0):
        model_filename_base = os.path.join(
            ".",
            self.results_dirname,
            self.results_filename_base + "_run" + str(run_id)
        )
        if not os.path.exists(model_filename_base):
            os.makedirs(model_filename_base)

        optimizer = torch.optim.Adam(self.model[run_id].parameters(), 1e-4)
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        post_pred = Compose(
            [EnsureType(), AsDiscrete(argmax=True, to_onehot=self.num_classes)]
        )
        post_label = Compose(
            [EnsureType(), AsDiscrete(to_onehot=self.num_classes)]
        )

        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []

        for epoch in range(self.max_epochs):
            print("-" * 10)
            print(f"{self.vfold_num}: epoch {epoch + 1}/{self.max_epochs}",flush=True)
            self.model[run_id].train()
            epoch_loss = 0
            epoch_size = 0
            for step,batch_data in enumerate(self.train_loader):
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model[run_id](inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_size = step
                print(
                    f"{step} /"
                    f" {len(self.train_files[self.vfold_num])//self.train_loader.batch_size},"
                    f" train_loss: {loss.item():.4f}", flush=True
                )
            epoch_loss /= epoch_size
            epoch_loss_values.append(epoch_loss)
            print(
                f"{self.vfold_num} epoch {epoch+1}" f" average loss: {epoch_loss:.4f}",
                flush=True,
            )

            if self.validation_interval > 0 and (epoch + 1) % self.validation_interval == 0:
                self.model[run_id].eval()
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs, val_labels = (val_data["image"], val_data["label"])
                        val_inputs = val_inputs.to(self.device)
                        val_labels = val_labels.to(self.device)
                        roi_size = (self.size_x, self.size_y)
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, self.batch_size_val, self.model[run_id]
                        )
                        # val_outputs = self.model[run_id](val_inputs)
                        val_outputs = [
                            post_pred(i) for i in decollate_batch(val_outputs)
                        ]
                        val_labels = [
                            post_label(i) for i in decollate_batch(val_labels)
                        ]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    metric_window = 5
                    metric_values.append(metric)
                    if epoch>100 and len(metric_values)>metric_window+1:
                        mean_metric = np.mean(metric_values[-metric_window:])
                        if mean_metric > best_metric:
                            best_metric = mean_metric
                            best_metric_epoch = epoch + 1
                            torch.save(
                                self.model[run_id].state_dict(),
                                os.path.join(model_filename_base,
                                             "best_model_" + str(self.vfold_num) + ".pth"),
                            )
                            print("saved new best metric model")
                    print(
                        f"Current epoch: {epoch + 1}"
                        f" current mean dice: {metric:.4f}"
                    )
                    print(
                        f"Best mean dice: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                    torch.save(
                        self.model[run_id].state_dict(),
                        os.path.join(model_filename_base,  "last_model_" + str(self.vfold_num) + ".pth"),
                    )
                    np.save(
                        os.path.join(model_filename_base, "loss_" + str(self.vfold_num) + ".npy"),
                        epoch_loss_values,
                    )
                    np.save(
                        os.path.join(model_filename_base, "val_dice_" + str(self.vfold_num) + ".npy"),
                        metric_values,
                    )
            if self.randomize_folds and self.refold_interval > 0 and (epoch + 1) % self.refold_interval == 0:
                self.setup_vfold_files()
                self.setup_training_vfold(self.vfold_num)

    def test_vfold(self, model_type="best", run_id=0, model_vfold=-1):
        if model_vfold == -1:
            model_vfold = self.vfold_num
        model_filename_base = os.path.join(
            ".",
            self.results_dirname,
            self.results_filename_base + "_run" + str(run_id)
        )

        model_file = os.path.join(
            model_filename_base,
            model_type + "_model_" + str(model_vfold) + ".pth"
        )

        test_outputs_total = []
        test_images_total = []
        test_labels_total = []
        test_filenames_total = []
        metric_total = 0

        if os.path.exists(model_file):
            self.model[run_id].load_state_dict(torch.load(model_file, map_location=self.device))
            self.model[run_id].eval()

            post_pred = Compose(
                [EnsureType(), AsDiscrete(argmax=True, to_onehot=self.num_classes)]
            )
            post_label = Compose(
                [EnsureType(), AsDiscrete(to_onehot=self.num_classes)]
            )
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            metric_total = 0
            metric_count = 0
            test_filenames_total = [ os.path.basename( test_file["image"] )
                                    for test_file in list(self.test_files[self.vfold_num][:]) ]
            with torch.no_grad():
                for batch_num, test_data in enumerate(self.test_loader):
                    roi_size = (self.size_x, self.size_y)
                    test_inputs, test_labels = (test_data["image"], test_data["label"])
                    test_inputs = test_inputs.to(self.device)
                    test_outputs = sliding_window_inference(
                        test_inputs,
                        roi_size,
                        self.batch_size_test,
                        self.model[run_id],
                    ).cpu()
                    # val_outputs = self.model[run_id](val_inputs)
                    tmp_outputs = [
                        post_pred(i) for i in decollate_batch(test_outputs)
                    ]
                    tmp_labels = [
                        post_label(i) for i in decollate_batch(test_labels)
                    ]
                    # compute metric for current iteration
                    dice_metric(y_pred=tmp_outputs, y=tmp_labels)
                    if batch_num == 0:
                        test_images_total = test_data["image"]
                        test_labels_total = test_data["label"]
                        test_outputs_total = test_outputs
                    else:
                        test_images_total = np.concatenate(
                            (test_images_total, test_data["image"]), axis=0
                        )
                        test_labels_total = np.concatenate(
                            (test_labels_total, test_data["label"]), axis=0
                        )
                        test_outputs_total = np.concatenate(
                            (test_outputs_total, test_outputs), axis=0
                        )
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_total += metric
                metric_count += 1
            metric_total /= metric_count
        else:
            print("ERROR: Model file not found:", model_file, "!!")

        return test_filenames_total, test_images_total, test_labels_total, test_outputs_total,metric_total

    def classify_vfold(self, model_type="best", run_ids=[0], model_vfold=-1):
        if model_vfold == -1:
            model_vfold = self.vfold_num
            
        test_filenames = []
        test_inputs = []
        test_ideal_outputs = []
        test_run_outputs = []
        for run_num,run_id in enumerate(run_ids):
            filenames, imgs, lbls, outs, metric = self.test_vfold(model_type, run_id, model_vfold)
            if run_num == 0:
                test_filenames = filenames
                test_inputs = imgs
                test_ideal_outputs = lbls
            test_run_outputs.append(outs)

        num_runs = len(test_run_outputs)
        num_images = len(test_run_outputs[0])

        prob_shape = test_run_outputs[0][0].shape

        test_ensemble_outputs = []
        prob = np.empty(prob_shape)
        for image_num in range(num_images):
            prob_total = np.zeros(prob_shape)
            for run_num in range(num_runs):
                run_output = test_run_outputs[run_num][image_num]
                prob = self.clean_probabilities_array(run_output)
                prob_total += prob
            prob_total /= num_runs
            prob = self.clean_probabilities_array(prob_total, use_blur=False)
            class_array = self.classify_probabilities_array(prob_total)
            test_ensemble_outputs.append(class_array)

        return test_filenames, test_inputs, test_ideal_outputs, test_ensemble_outputs

    def verify_vfold(self):
        batch = monai.utils.misc.first(self.train_loader)
        im = batch["image"]
        label = batch["label"]
        print(type(im), im.shape, label, label.shape)
        
    def view_training_image(self, image_num=0):
        img_name = self.all_train_images[image_num]
        img = itk.imread(img_name)
        lbl_name = self.all_train_labels[image_num]
        lbl = itk.imread(lbl_name)
        print(str(image_num), img_name, lbl_name)
        num_plots = 5
        num_slices = img.shape[0]
        step_slices = num_slices / num_plots
        plt.figure(figsize=[20, 10])
        for s in range(num_plots):
            slice_num = int(step_slices * s)
            plt.subplot(2, num_plots, s + 1)
            plt.axis('off')
            plt.imshow(img[slice_num, :, :])
            plt.subplot(2, num_plots, num_plots + s + 1)
            plt.axis('off')
            plt.imshow(lbl[slice_num, :, :])

    def view_training_vfold_batch(self, batch_num=0):
        with torch.no_grad():
            for count, batch_data in enumerate(self.train_loader):
                if count == batch_num:
                    inputs, labels = (batch_data["image"], batch_data["label"])
                    num_images = inputs.shape[0]
                    for i in range(num_images):
                        img = inputs[i]
                        lbl = labels[i]
                        num_channels = img.shape[0]
                        plt.figure(figsize=[30, 30])
                        for c in range(num_channels):
                            plt.subplot(
                                num_images,
                                num_channels + 1,
                                i * (num_channels + 1) + c + 1,
                            )
                            plt.axis('off')
                            plt.imshow(rotate(img[c, :, :],270))
                        plt.subplot(
                            num_images,
                            num_channels + 1,
                            i * (num_channels + 1) + num_channels + 1,
                        )
                        plt.axis('off')
                        plt.imshow(rotate(lbl[0, :, :],270))
                        plt.show()
                    break

    def view_training_metric_curves(self, vfold_num, run_id=0):
        model_filename_base = os.path.join(
            ".",
            self.results_dirname,
            self.results_filename_base + "_run" + str(run_id)
        )
        loss_file = os.path.join(model_filename_base, "loss_" + str(vfold_num) + ".npy")
        if os.path.exists(loss_file):
            plt.figure("Train", (12, 6))
            epoch_loss_values = np.load(loss_file)
            plt.subplot(1, 2, 1)
            plt.title("Epoch Average Loss")
            x = [i + 1 for i in range(len(epoch_loss_values))]
            y = epoch_loss_values
            plt.xlabel("epoch")
            plt.plot(x, y)
            plt.ylim([0.0, 1.0])

            metric_file = os.path.join(model_filename_base, "val_dice_" + str(vfold_num) + ".npy")
            if os.path.exists(metric_file):
                metric_values = np.load(metric_file)

                plt.subplot(1, 2, 2)
                plt.title("Val Mean Dice")
                x = [2 * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                plt.xlabel("epoch")
                plt.plot(x, y)
                plt.ylim([0.0, 1.0])

            else:
                print("ERROR: Cannot read metric file:", metric_file)

            plt.show()
        else:
            print("ERROR: Cannot read metric file:", loss_file)

    def view_testing_results_vfold(self, model_type="best", run_ids=[0], model_vfold=-1, summary_only=False):
        print("VFOLD =", self.vfold_num, "of", self.num_folds - 1)
        if model_vfold == -1:
            model_vfold = self.vfold_num
        else:
            print("   Using model from vfold", model_vfold)

        test_inputs_image_filenames = []
        test_inputs_images = []
        test_ideal_outputs = []
        test_net_outputs = []
        test_metric = 0
        test_metric_count = 0
        for run_num,run_id in enumerate(run_ids):
            run_input_image_filenames, run_input_images, run_ideal_outputs, run_net_outputs, run_metric = self.test_vfold(
                model_type, run_id, model_vfold)
            test_metric += run_metric
            test_metric_count += 1
            test_net_outputs.append(run_net_outputs)
            if run_num == 0:
                test_input_image_filenames = run_input_image_filenames
                test_input_images = run_input_images
                test_ideal_outputs = run_ideal_outputs
        test_metric /= test_metric_count

        num_runs = len(run_ids)

        if not summary_only:
            num_subplots = max(
                self.net_in_channels + 1,
                self.num_classes + self.num_classes + 2
            )

        for image_num in range(len(test_input_images)):
            fname = os.path.basename( test_input_image_filenames[image_num] )
            if not summary_only:
                print("Image:", fname)
                width = 3 * (self.net_in_channels+1)
                height = 3 * (num_runs+1)+0.25
                plt.figure("Testing", figsize=(width,height))
                subplot_num = 1
                for c in range(self.net_in_channels):
                    plt.subplot(num_runs+1, num_subplots, subplot_num)
                    plt.title(f"F"+str(c))
                    tmpV = test_input_images[image_num, c, :, :]
                    plt.axis('off')
                    plt.imshow(rotate(tmpV,270), cmap="gray")
                    subplot_num += 1
                plt.subplot(num_runs+1, num_subplots, num_subplots)
                plt.title(f"L")
                tmpV = test_ideal_outputs[image_num, 0, :, :]
                for c in range(self.num_classes):
                    tmpV[0, c] = c
                plt.axis('off')
                plt.imshow(rotate(tmpV,270))
                subplot_num += 1

            # run probabilities
            prob_shape = test_net_outputs[0][0].shape
            prob_total = np.zeros(prob_shape)
            run_output = np.empty(prob_shape)
            for run_num in range(num_runs):
                run_output = test_net_outputs[run_num][image_num]
                prob = self.clean_probabilities_array(run_output)
                prob_total += prob
                if not summary_only:
                    subplot_num = num_subplots*(run_num+1) + 2
                    for c in range(self.num_classes):
                        plt.subplot(num_runs+1, num_subplots, subplot_num)
                        plt.title(f"R" + str(run_num) + " C" + str(c))
                        tmpV = prob[c]
                        plt.axis('off')
                        plt.imshow(rotate(tmpV,270), cmap="gray")
                        subplot_num += 1
            prob_total /= num_runs

            # ensemble probabilities
            prob = self.clean_probabilities_array(prob_total, use_blur=False)
            if not summary_only:
                subplot_num = (num_runs+1)*num_subplots - self.num_classes
                for c in range(self.num_classes):
                    plt.subplot(num_runs+1, num_subplots, subplot_num)
                    plt.title(f"E"+str(c))
                    tmpV = prob[c]
                    plt.axis('off')
                    plt.imshow(rotate(tmpV,270), cmap="gray")
                    subplot_num += 1

            # ensemble classifications
            class_array = self.classify_probabilities_array(prob)

            #itk.imwrite(
                #itk.GetImageFromArray(class_array.astype(np.float32)),
                #"class_image_final_f" + str(self.vfold_num) +
                #"i" + str(image_num) + ".mha" )
            if not summary_only:
                plt.subplot(num_runs+1, num_subplots, subplot_num)
                plt.title(f"Label")
                tmpV = class_array
                for c in range(self.num_classes):
                    tmpV[0, c] = c
                plt.axis('off')
                plt.imshow(rotate(tmpV,270))
                plt.show()

                class_image = itk.GetImageFromArray(
                    class_array.astype(np.float32))
                itk.imwrite(class_image, os.path.join(
                    ".",
                    self.results_dirname,
                    os.path.splitext(fname)[0]+".out.mha" ))
        print( "  Test Mean Dice Score =", test_metric )
