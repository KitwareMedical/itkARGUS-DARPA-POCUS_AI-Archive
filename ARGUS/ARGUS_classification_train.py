import warnings
warnings.filterwarnings("ignore")

import monai
from monai.utils import first, set_determinism
from monai.transforms import (
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandFlipd,
    RandZoomd,
    RandRotated,
    Resized,
    ToTensord,
)
from monai.data import (
    PersistentDataset,
    CacheDataset,
    DataLoader,
    Dataset,
    decollate_batch,
    list_data_collate,
)

import torch

import os
import json
from glob import glob

import configparser

import random

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

import itk
from itk import TubeTK as tube

from ARGUS_Transforms import ARGUS_RandSpatialCropSlicesd
from ARGUS_classification_inference import ARGUS_classification_inference

class ARGUS_classification_train(ARGUS_classification_inference):
    def __init__(self, config_file_name, network_name="vfold", device_num=0):
        super().__init__(config_file_name, network_name, device_num)
        
        config = configparser.ConfigParser()
        config.read(config_file_name)

        self.image_dirname = json.loads(config[network_name]['image_dirname'])
        self.image_filesuffix = config[network_name]['image_filesuffix']
        
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
        
        self.class_fileprefix = json.loads(config[network_name]['class_fileprefix'])
        self.class_labels = json.loads(config[network_name]['class_labels'])

        self.results_filename_base = config[network_name]['results_filename_base']
        self.results_dirname = config[network_name]['results_dirname']
        
        tmp_str = config[network_name]['use_persistent_cache']
        self.use_persistent_cache = False
        if tmp_str == "True":
            self.use_persistent_cache = True
        
        self.max_epochs = int(config[network_name]['max_epochs'])
        
        tmp_str = config[network_name]['already_preprocessed']
        self.already_preprocessed = False
        if tmp_str == "True":
            self.already_preprocessed = True

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

        if self.already_preprocessed:
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AsChannelFirstd(keys=["image"]),
                    RandFlipd(prob=0.5, spatial_axis=1, keys=["image"]),
                    RandZoomd(prob=0.5, 
                        min_zoom=1.0,
                        max_zoom=1.1,
                        keep_size=True,
                        mode=['bilinear'],
                        keys=['image'],
                    ),ToTensord(keys=["image"], dtype=torch.float),
                ]
            )
            self.val_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AsChannelFirstd(keys=["image"]),
                    ToTensord(keys=["image"], dtype=torch.float),
                ]
            )
            self.test_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AsChannelFirstd(keys=["image"]),
                    ToTensord(keys=["image"], dtype=torch.float),
                ]
            )
        else:
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AsChannelFirstd(keys=["image"]),
                    Resized(
                        spatial_size=[self.size_y,self.size_x],
                        mode=['bilinear'],
                        keys=["image"],
                    ),
                    RandRotated(prob=0.2,
                        range_z=0.15,
                        keep_size=True,
                        keys=['image'],
                    ),
                    ARGUS_RandSpatialCropSlicesd(
                        num_slices=self.num_slices,
                        axis=0,
                        reduce_to_statistics=self.reduce_to_statistics,
                        extended=self.reduce_to_statistics,
                        include_center_slice=self.reduce_to_statistics,
                        include_gradient=self.reduce_to_statistics,
                        keys=["image"],
                    ),
                    RandFlipd(prob=0.5, spatial_axis=0, keys=["image"]),
                    RandZoomd(prob=0.5, 
                        min_zoom=1.0,
                        max_zoom=1.1,
                        keep_size=True,
                        mode=['bilinear'],
                        keys=['image'],
                    ),ToTensord(keys=["image"], dtype=torch.float),
                ]
            )
            self.val_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AsChannelFirstd(keys=["image"]),
                    Resized(
                        spatial_size=[self.size_y,self.size_x],
                        mode=['bilinear'],
                        keys=["image"],
                    ),
                    ARGUS_RandSpatialCropSlicesd(
                        num_slices=self.num_slices,
                        axis=0,
                        reduce_to_statistics=self.reduce_to_statistics,
                        extended=self.reduce_to_statistics,
                        include_center_slice=self.reduce_to_statistics,
                        include_gradient=self.reduce_to_statistics,
                        keys=["image"],
                    ),
                    ToTensord(keys=["image"], dtype=torch.float),
                ]
            )
            self.test_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AsChannelFirstd(keys=["image"]),
                    Resized(
                        spatial_size=[self.size_y,self.size_x],
                        mode=['bilinear'],
                        keys=["image"],
                    ),
                    ARGUS_RandSpatialCropSlicesd(
                        num_slices=self.num_slices,
                        center_slice=self.testing_slice,
                        axis=0,
                        reduce_to_statistics=self.reduce_to_statistics,
                        extended=self.reduce_to_statistics,
                        include_center_slice=self.reduce_to_statistics,
                        include_gradient=self.reduce_to_statistics,
                        keys=["image"],
                    ),
                    ToTensord(keys=["image"], dtype=torch.float),
                ]
            )

    def init_model(self, model_num):
        self.model[model_num] = monai.networks.nets.DenseNet121(
            spatial_dims=self.net_in_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
        ).to(self.device)

    def setup_vfold_files(self):
        all_train_images = []
        for dirname in self.image_dirname:
            all_train_images = all_train_images + sorted(glob(os.path.join(dirname, self.image_filesuffix)))
        self.class_train_images = []
        self.class_train_labels = []
        for i,pre in enumerate(self.class_fileprefix):
            class_files = [x for x in all_train_images if pre in os.path.basename(x)[:len(pre)]]
            class_labels = [i] * len(class_files)
            self.class_train_images.append(class_files)
            self.class_train_labels.append(class_labels)
            print(f"{pre} : num images = {len(class_files)}")
        self.all_train_images = [i for imglist in self.class_train_images for i in imglist]
        self.all_train_labels = [i for lbllist in self.class_train_labels for i in lbllist]
        print( f"Total num images = {len(self.all_train_images)}" )
        
        num_images = len(self.all_train_images)
        print("Num images / labels =", num_images, len(self.all_train_labels))

        self.all_train_data = list(zip(self.all_train_images, self.all_train_labels))

        if self.randomize_folds==True:
            random.shuffle(self.all_train_data)
            
        fold_data = []
        fold_size = num_images // self.num_folds
        extra_case = num_images - (fold_size * self.num_folds)
        count = 0
        for i in range(self.num_folds):
            fsize = fold_size
            if i<extra_case:
                fsize += 1
            fold_data.append(self.all_train_data[count:count+fsize])
            count += fsize

        #for i in range(self.num_folds):
            #print(f"VFold-data[{i}] = {fold_data[i]}")

        self.train_files = []
        self.val_files = []
        self.test_files = []

        for i in range(self.num_folds):
            tr_folds = []
            va_folds = []
            te_folds = []
            if self.num_folds == 1:
                if self.train_data_portion == 1.0:
                    tr_folds = fold_data[0]
                    te_folds = fold_data[0]
                    va_folds = fold_data[0]
                else:
                    num_pre = len(fold_data[0])
                    num_tr = int(num_pre * self.train_data_portion)
                    num_va = int(num_pre * self.validation_data_portion)
                    num_te = int(num_pre * self.test_data_portion)
                    if self.test_data_portion > 0 and num_te < 1 and num_tr > 2:
                        num_tr -= 1
                        num_te = 1
                    if self.validation_data_portion > 0 and num_va < 1 and num_tr > 2:
                        num_tr -= 1
                        num_va = 1
                    tr_folds = list(fold_data[0][0:num_tr])
                    if num_va>0:
                        va_folds = list(fold_data[0][num_tr:num_tr+num_va])
                    if num_te>0:
                        te_folds = list(fold_data[0][num_tr+num_va:])
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
                    tr_folds.extend(fold_data[f % self.num_folds])
                if num_va > 0:
                    for f in range(i + num_tr, i + num_tr + num_va):
                        va_folds.extend(fold_data[f % self.num_folds])
                if num_te > 0:
                    for f in range(i + num_tr + num_va, i + num_tr + num_va + num_te):
                        te_folds.extend(fold_data[f % self.num_folds])
            self.train_files.append( [
                {"image": folddata[0], "label": folddata[1]}
                for folddata in tr_folds
                ] )
            if len(va_folds) > 0:
                self.val_files.append( [
                    {"image": folddata[0], "label": folddata[1]}
                    for folddata in va_folds
                    ] )
            if len(te_folds) > 0:
                self.test_files.append( [
                    {"image": folddata[0], "label": folddata[1]}
                    for folddata in te_folds
                    ] )
                    
            print( f"**** VFold =", i, ":", len(self.train_files[i]),
                 len(self.val_files[i]), len(self.test_files[i]))
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

        loss_function = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(self.model[run_id].parameters(), 1e-4)
        
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
                
                num_correct = 0.0
                metric_count = 0
                metric_value = 0
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs = val_data["image"].to(self.device)
                        val_labels = val_data["label"].to(self.device)
                        val_outputs = self.model[run_id](val_inputs)
                        out_labels = val_outputs.argmax(dim=1)
                        value = torch.eq(out_labels, val_labels)
                        metric_count += len(value)
                        num_correct += value.sum().item()
                        val_outputs2 = val_outputs.cpu()
                        for i in range(len(val_outputs2)):
                            minv = min(val_outputs2[i])
                            maxv = max(val_outputs2[i])
                            maxv2 = max([x for x in val_outputs2[i] if x < maxv])
                            denom = maxv2-minv
                            if denom == 0:
                                denom = 1
                            metric_value += (val_outputs2[i,val_labels[i]] - minv)/denom
                    correct = num_correct / metric_count
                    metric = (metric_value / metric_count) * correct

                    metric_values.append(metric)
                    
                    metric_window = 5
                    mean_metric = -1
                    if len(metric_values) > metric_window:
                        mean_metric = np.mean(metric_values[-metric_window:])
                    if epoch > self.max_epochs // 5:
                        if mean_metric > best_metric:
                            best_metric = mean_metric
                            best_metric_epoch = epoch + 1
                            torch.save(
                                self.model[run_id].state_dict(),
                                os.path.join(model_filename_base,
                                             "best_model_" + str(self.vfold_num) + ".pth"),
                            )
                            print("saved new best metric model")
                    print( f"Current epoch: {epoch + 1}" )
                    print( f"   Current portion correct: {correct:.4f}" )
                    print( f"   Current accuracy value: {metric:.4f}" )
                    print( f"   Current mean accuracy value: {mean_metric:.4f}" )
                    print( f"Best mean accuracy value: {best_metric:.4f}" )
                    print( f"    at epoch: {best_metric_epoch}" )
                    torch.save(
                        self.model[run_id].state_dict(),
                        os.path.join(model_filename_base,  "last_model_" + str(self.vfold_num) + ".pth"),
                    )
                    np.save(
                        os.path.join(model_filename_base, "loss_" + str(self.vfold_num) + ".npy"),
                        epoch_loss_values,
                    )
                    np.save(
                        os.path.join(model_filename_base, "val_acc_" + str(self.vfold_num) + ".npy"),
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

        if os.path.exists(model_file):
            self.model[run_id].load_state_dict(torch.load(model_file, map_location=self.device))
            self.model[run_id].eval()

            test_filenames_total = [ os.path.basename( test_file["image"] )
                                    for test_file in list(self.test_files[self.vfold_num][:]) ]
            
            num_correct = 0.0
            metric = 0.0
            metric_count = 0
            with torch.no_grad():
                for batch_num, test_data in enumerate(self.test_loader):
                    test_inputs = test_data["image"].to(self.device)
                    test_labels = test_data["label"].to(self.device)
                    test_outputs = self.model[run_id](test_inputs)
                    out_labels = test_outputs.argmax(dim=1)
                    
                    value = torch.eq(out_labels, test_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                    for i in range(len(out_labels)):
                        maxv = max(test_outputs[i])
                        maxv2 = max([x for x in test_outputs[i] if x < maxv])
                        metric += test_outputs[i, out_labels[i]] - maxv2

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
                        
                accuracy = num_correct / metric_count
                metric = metric / metric_count
                metric *= accuracy
                print( f"   Accuracy = {accuracy}" )
                print( f"   Confidence = {metric}" )
                
        else:
            print("ERROR: Model file not found:", model_file, "!!")

        return test_filenames_total, test_images_total, test_labels_total, test_outputs_total, metric
    
    def classify_vfold(self, model_type="best", run_ids=[0], model_vfold=-1):
        if model_vfold == -1:
            model_vfold = self.vfold_num
            
        test_filenames = []
        test_inputs = []
        test_ideal_outputs = []
        test_run_outputs = []
        for run_num,run_id in enumerate(run_ids):
            filenames, imgs, lbls, outs = self.test_vfold(model_type, run_id, model_vfold)
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
                prob = self.clean_probabilities(run_output)
                prob_total += prob
            prob_total /= num_runs
            prob = self.clean_probabilities(prob_total, use_blur=False)
            class_array = self.classify_probabilities(prob_total)
            test_ensemble_outputs.append(class_array)

        return test_filenames, test_inputs, test_ideal_outputs, test_ensemble_outputs

    def verify_vfold(self):
        batch = monai.utils.misc.first(self.train_loader)
        im = batch["image"]
        label = batch["label"]
        print(type(im), im.shape, label, label.shape)

    def view_training_image(self, image_num=0):
        img_name = self.all_train_images[image_num]
        print(img_name)
        img = itk.imread(img_name)
        lbl = self.all_train_labels[image_num]
        if img.shape[0]>12:
            num_plots = 5
        else:
            num_plots = img.shape[0]
        num_slices = img.shape[0]
        step_slices = num_slices / num_plots
        plt.figure(figsize=[20, 10])
        for s in range(num_plots):
            slice_num = int(step_slices * s)
            plt.subplot(1, num_plots, s + 1)
            plt.axis('off')
            plt.imshow(img[slice_num, :, :])
        plt.show()
        print(f"Label = {lbl}")

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
                        plt.show()
                        print(f"Label = {lbl}")
                    break

    def view_metric_curves(self, vfold_num, run_id=0):
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

            metric_file = os.path.join(model_filename_base, "val_acc_" + str(vfold_num) + ".npy")
            if os.path.exists(metric_file):
                metric_values = np.load(metric_file)

                plt.subplot(1, 2, 2)
                plt.title("Val Mean Accuracy")
                x = [2 * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                plt.xlabel("epoch")
                plt.plot(x, y)

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
        confidence = 0.0
        confidence_count = 0
        for run_num,run_id in enumerate(run_ids):
            run_input_image_filenames, run_input_images, run_ideal_outputs, run_net_outputs, run_metric = self.test_vfold(
                model_type, run_id, model_vfold)
            test_net_outputs.append(run_net_outputs)
            confidence += run_metric
            confidence_count += 1
            if run_num == 0:
                test_input_image_filenames = run_input_image_filenames
                test_input_images = run_input_images
                test_ideal_outputs = run_ideal_outputs
        confidence /= confidence_count
        
        num_runs = len(run_ids)

        test_metric = 0
        test_metric_count = 0
        for image_num in range(len(test_input_images)):
            fname = os.path.basename( test_input_image_filenames[image_num] )
            if not summary_only:
                print("Image:", fname)
                plt.figure("Testing")
                for c in range(self.net_in_channels):
                    plt.subplot(1, self.net_in_channels, c+1)
                    plt.title(f"F"+str(c))
                    tmpV = test_input_images[image_num, c, :, :]
                    plt.axis('off')
                    plt.imshow(rotate(tmpV,270), cmap="gray")
                plt.show()
                print( "   Ideal output =", test_ideal_outputs[image_num] )

            # run probabilities
            prob_shape = test_net_outputs[0][0].shape
            prob_total = np.zeros(prob_shape)
            run_output = np.empty(prob_shape)
            for run_num in range(num_runs):
                run_output = test_net_outputs[run_num][image_num]
                prob = self.clean_probabilities(run_output)
                prob_total += prob
                if not summary_only:
                    print( f"   Run output ({run_num}) = {run_output}" )
            prob_total /= num_runs

            # ensemble probabilities
            prob = self.clean_probabilities(prob_total)
            if not summary_only:
                print( f"   Ensemble output = {prob}" )

            # ensemble classifications
            class_array = self.classify_probabilities(prob)

            if not summary_only:
                print( f"   Final output = {class_array}" )
                print( "" )
            test_metric += abs(class_array - test_ideal_outputs[image_num])
            test_metric_count += 1.0
        test_metric /= test_metric_count
        print("   Accuracy =", 1.0 - test_metric)
        print( f"   Confidence = {confidence}" )
