#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings 
warnings.filterwarnings("ignore")

from monai.transforms import (
    Activations,
    AddChannel,
    AsChannelFirst,
    AsDiscrete,
    Compose,
    EnsureType,
    LoadImage,
    RandFlip,
    RandSpatialCrop,
    RandZoom,
    Resize,
    ScaleIntensity,
    ScaleIntensityRange,
    SpatialCrop,
    ToTensor,
)
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
import monai.utils as utils

import torch

import matplotlib.pyplot as plt

import os
from glob import glob

import sys

import numpy as np

import itk

import site
site.addsitedir('../../ARGUS')
from ARGUSUtils_Transforms import *


# In[5]:


img1_dir = "../../Data/VFoldData/ROIData/"

all_images = sorted(glob(os.path.join(img1_dir, '*Class[NS]*.roi.nii.gz')))
all_labels = sorted(glob(os.path.join(img1_dir, '*Class[NS]*.roi.overlay.nii.gz')))

num_classes = 2

num_slices = 48

num_workers_tr = 24
batch_size_tr = 24
num_workers_vl = 24
batch_size_vl = 8

model_filename_base = "BAMC_PTX_2DROI_DenseNet-2Class.best_model.vfold"

num_images = len(all_images)

print(num_images)

if len(sys.argv) == 3:
    device_num = int(sys.argv[1])
    num_devices = int(sys.argv[2])
    print("Using device", str(device_num),"of", str(num_devices))
else:
    print("Device number assumed to be 0")
    device_num = 0
    num_devices = 1

num_folds = 15

ns_prefix = ['025ns','026ns','027ns','035ns','048ns','055ns','117ns',
             '135ns','193ns','210ns','215ns','218ns','219ns','221ns','247ns']
s_prefix = ['004s','019s','030s','034s','037s','043s','065s','081s',
            '206s','208s','211s','212s','224s','228s','236s','237s']

fold_prefix_list = []
fold_label_list = []
ns_count = 0
s_count = 0
for i in range(num_folds):
    if i%2 == 0:
        num_ns = 1
        num_s = 1
        if i > num_folds-3:
            num_s = 2
    else:
        num_ns = 1
        num_s = 1
    f = []
    for ns in range(num_ns):
        f.append([ns_prefix[ns_count+ns]])
    ns_count += num_ns
    for s in range(num_s):
        f.append([s_prefix[s_count+s]])
    s_count += num_s
    fold_prefix_list.append(f)
        
train_files = []
train_labels = []
val_files = []
val_labels = []
test_files = []
test_labels = []
for i in range(num_folds):
    tr_folds = []
    for f in range(i,i+num_folds-2):
        tr_folds.append(fold_prefix_list[f%num_folds])
    tr_folds = list(np.concatenate(tr_folds).flat)
    va_folds = list(np.concatenate(fold_prefix_list[(i+num_folds-2) % num_folds]).flat)
    te_folds = list(np.concatenate(fold_prefix_list[(i+num_folds-1) % num_folds]).flat)
    img = [im for im in all_images if any(pref in im for pref in tr_folds)]
    seg = []
    for im in img:
        if "ClassN" in im:
            seg.append(0)
        else:
            seg.append(1)
    train_files.append(img)
    train_labels.append(seg)
    img = [im for im in all_images if any(pref in im for pref in va_folds)]
    seg = []
    for im in img:
        if "ClassN" in im:
            seg.append(0)
        else:
            seg.append(1)
    val_files.append(img)
    val_labels.append(seg)
    img = [im for im in all_images if any(pref in im for pref in te_folds)]
    seg = []
    for im in img:
        if "ClassN" in im:
            seg.append(0)
        else:
            seg.append(1)
    test_files.append(img)
    test_labels.append(seg)
    print(len(train_files[i]),len(val_files[i]),len(test_files[i]))
    print(len(train_labels[i]),len(val_labels[i]),len(test_labels[i]))


# In[6]:


train_transforms = Compose(
    [
        LoadImage(image_only=True),
        AsChannelFirst(),
        ARGUS_RandSpatialCropSlices(
            num_slices=num_slices,
            axis=0,
            reduce_to_statistics=True),
        ScaleIntensity(
            channel_wise=True),
        RandFlip(prob=0.5, 
            spatial_axis=1),
        RandZoom(prob=0.5, 
            min_zoom=1.0,
            max_zoom=1.2,
            keep_size=True,
            mode='bilinear'),
        EnsureType(),
    ]
)

val_transforms = Compose(
    [
        LoadImage(image_only=True), 
        AsChannelFirst(),
        ARGUS_RandSpatialCropSlices(
            num_slices=num_slices,
            axis=0,
            center_slice=30,
            reduce_to_statistics=True),
        ScaleIntensity(), 
        EnsureType()
    ]
)

y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=num_classes)])


# In[7]:


class PTXDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


# In[8]:


train_ds = [PTXDataset(train_files[i], train_labels[i], train_transforms) for i in range(num_folds)]
train_loader = [torch.utils.data.DataLoader(train_ds[i], batch_size=batch_size_tr, shuffle=True, num_workers=num_workers_tr) 
                for i in range(num_folds)]

val_ds = [PTXDataset(val_files[i], val_labels[i], train_transforms) for i in range(num_folds)]
val_loader = [torch.utils.data.DataLoader(val_ds[i], batch_size=batch_size_vl, num_workers=num_workers_vl)
              for i in range(num_folds)]


# In[9]:


imgnum = 2
img, lbl = utils.first(train_loader[0])


# In[10]:


print(lbl[0])
plt.subplots()
plt.imshow(img[imgnum,0,:,:])
plt.subplots()
plt.imshow(img[imgnum,1,:,:])
print("Data Size =", img.shape)
roi_size = img.shape[2:]
print("ROI Size =", roi_size)


# In[11]:


device = torch.device("cuda:"+str(device_num))

def vfold_train(vfold_num, train_loader, val_loader):
    model = DenseNet121(spatial_dims=2, in_channels=2,
                        out_channels=num_classes).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric()

    max_epochs = 1000
    val_interval = 2
    
    best_metric = num_classes*100
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"{vfold_num}: epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
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
                    vimages, vlabels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(vimages)], dim=0)
                    y = torch.cat([y, vlabels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                diff = 0
                for i in range(len(y_onehot)):
                    for c in range(num_classes):
                        diff += (y_onehot[i][c] - y_pred_act[i][c])**2
                result = float(diff)
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if epoch>100 and result <= best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                        model_filename_base+'_'+str(vfold_num)+'.pth')
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current SSD: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best SSD: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                    )

    np.save(model_filename_base+"_loss_"+str(vfold_num)+".npy", epoch_loss_values)
    np.save(model_filename_base+"_auc_"+str(vfold_num)+".npy", metric_values)

# In[ ]:


for i in range(device_num,num_folds,num_devices):
    vfold_train(i, train_loader[i], val_loader[i])


# In[ ]:


# In[ ]:




