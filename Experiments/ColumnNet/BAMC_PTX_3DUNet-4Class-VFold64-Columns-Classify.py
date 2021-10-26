#!/usr/bin/env python
# coding: utf-8

# In[2]:


from monai.utils import first, set_determinism
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    RandFlip,
    RandSpatialCrop,
    RandZoom,
    ScaleIntensityRange,
    SpatialCrop,
    ToTensor,
)
from monai.handlers.utils import from_engine
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
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

import site
site.addsitedir('../../ARGUS')
from ARGUSUtils_Transforms import *


# In[3]:


img1_dir = "../../Data/VFoldData/ColumnData/"

all_images = sorted(glob(os.path.join(img1_dir, '*_Class?_*.mha')))

num_folds = 10

num_classes = 3

num_workers_tr = 8
batch_size_tr = 8
num_workers_va = 8
batch_size_va = 2

model_filename_base = "BAMC_PTX_3DUNet-4Class.best_model.vfold64-Columns-Class"

num_images = len(all_images)
print(num_images)


# In[14]:


ns_prefix = ['025ns','026ns','027ns','035ns','048ns','055ns','117ns',
             '135ns','193ns','210ns','215ns','218ns','219ns','221ns','247ns']
s_prefix = ['004s','019s','030s','034s','037s','043s','065s','081s',
            '206s','208s','211s','212s','224s','228s','236s','237s']

fold_prefix_list = []
ns_count = 0
s_count = 0
for i in range(num_folds):
    if i%2 == 0:
        num_ns = 2
        num_s = 1
        if i > num_folds-3:
            num_s = 2
    else:
        num_ns = 1
        num_s = 2
    f = []
    for ns in range(num_ns):
        f.append([ns_prefix[ns_count+ns]])
    ns_count += num_ns
    for s in range(num_s):
        f.append([s_prefix[s_count+s]])
    s_count += num_s
    fold_prefix_list.append(f)
        
train_files = []
train_classes = []
val_files = []
val_classes = []
test_files = []
test_classes = []

for i in range(num_folds):
    tr_folds = []
    for f in range(i,i+num_folds-2):
        tr_folds.append(fold_prefix_list[f%num_folds])
    tr_folds = list(np.concatenate(tr_folds).flat)
    va_folds = list(np.concatenate(fold_prefix_list[(i+num_folds-2) % num_folds]).flat)
    te_folds = list(np.concatenate(fold_prefix_list[(i+num_folds-1) % num_folds]).flat)

    train_files.append([im for im in all_images if any(pref in im for pref in tr_folds)])
    fold_classes = []
    for file in train_files[len(train_files)-1]:
        if 'ClassN' in file:
            fold_classes.append([0])
        elif 'ClassR' in file:
            fold_classes.append([1])
        elif 'ClassS' in file:
            fold_classes.append([2])
        else:
            print("Error: Class tag not found in validation file", file)
    train_classes.append(fold_classes)

    val_files.append([im for im in all_images if any(pref in im for pref in va_folds)])
    fold_classes = []
    for file in val_files[len(val_files)-1]:
        if 'ClassN' in file:
            fold_classes.append([0])
        elif 'ClassR' in file:
            fold_classes.append([1])
        elif 'ClassS' in file:
            fold_classes.append([2])
        else:
            print("Error: Class tag not found in validation file", file)
    val_classes.append(fold_classes)

    test_files.append([im for im in all_images if any(pref in im for pref in te_folds)])
    fold_classes = []
    for file in test_files[len(test_files)-1]:
        if 'ClassN' in file:
            fold_classes.append([0])
        elif 'ClassR' in file:
            fold_classes.append([1])
        elif 'ClassS' in file:
            fold_classes.append([2])
        else:
            print("Error: Class tag not found in validation file", file)
    test_classes.append(fold_classes)

    print(len(train_files[i]),len(val_files[i]),len(test_files[i]))


# In[15]:


train_transforms = Compose(
    [
        LoadImage(),
        AddChannel(),
        ScaleIntensityRange(
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0),
        ARGUS_RandSpatialCropSlices(
            num_slices=48,
            axis=2),
        ARGUS_RandSpatialCropSlices(
            num_slices=8,
            axis=0,
            require_labeled=True),
        RandFlip(
            prob=0.5, 
            spatial_axis=2),
        RandFlip(
            prob=0.5, 
            spatial_axis=0),
        RandZoom(
            prob=0.5, 
            min_zoom=1.0,
            max_zoom=1.2,
            keep_size=True,
            mode='trilinear'),
        ToTensor(),
    ]
)

val_transforms = Compose(
    [
        LoadImage(),
        AddChannel(),
        ScaleIntensityRange(
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0),
        ARGUS_RandSpatialCropSlices(
            num_slices=48,
            axis=2),
        ARGUS_RandSpatialCropSlices(
            num_slices=8,
            axis=0,
            require_labeled=True),
        ToTensor(),
    ]
)

y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=num_classes)])


# In[16]:


class ColumnDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

print(str(num_folds), len(train_files), len(train_classes))
train_ds = [ColumnDataset(train_files[i], train_classes[i], transforms=train_transforms)
                for i in range(num_folds)]
train_loader = [torch.utils.data.DataLoader(train_ds[i], batch_size=batch_size_tr, shuffle=True)
                for i in range(num_folds)]

print(str(num_folds), len(val_files), len(val_classes))
val_ds = [ColumnDataset(val_files[i], val_classes[i], transforms=val_transforms)
                for i in range(num_folds)]
val_loader = [torch.utils.data.DataLoader(val_ds[i], batch_size=batch_size_va, shuffle=True)
                for i in range(num_folds)]


# In[23]:


# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:3")


# In[24]:


def vfold_train(vfold_num, train_loader, val_loader):
    model = DenseNet121(spatial_dims=3, in_channels=1,
                    out_channels=num_classes).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric()

    max_epochs = 1000
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    
    root_dir = "."

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"{vfold_num}: epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data[0].to(device),
                batch_data[1].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"{vfold_num} epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, model_filename_base+'_'+str(vfold_num)+'.pth'))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

    np.save(model_filename_base+"_loss_"+str(vfold_num)+".npy", epoch_loss_values)
    np.save(model_filename_base+"_acc_"+str(vfold_num)+".npy", metric_values)


# In[ ]:


for i in range(0,num_folds):
    vfold_train(i, train_loader[i], val_loader[i])

