from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from monai.transforms import Compose
import torchio as tio
from glob import glob
import os
import site
site.addsitedir('/home/local/KHQ/christopher.funk/code/AnatomicRecon-POCUS-AI/PTX/ARGUS')
from ARGUSUtils_Transforms import *

from src import utils

log = utils.get_logger(__name__)


class PTXDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

    To test: 
        >>> import src.datamodules.ptx_datamodule as ptx_d
        >>> d = ptx_d.PTXDataModule(data_dir="/data/krsdata2-pocus-ai-synced/root/Data_PTX/VFoldData/BAMC-PTX*Sliding-Annotations-Linear/")
        >>> d.prepare_data()
        >>> d.setup()
    """

    def __init__(
        self,
        data_dir: str = "data/PTX",
        train_val_ratio: float = 0.8,
        data_seed: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.subjects = None
        self.test_subjects = None

        # data transformations
        self.transforms = self.get_transforms

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 3

    def get_transforms(self):
        train_transforms = Compose(
            [
                # LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(
                    a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0,
                    keys=["image"]),
                SpatialCropd(
                    roi_start=[80,0,1],
                    roi_end=[240,320,61],
                    keys=["image", "label"]),
                ARGUS_RandSpatialCropSlicesd(
                    num_slices=num_slices,
                    axis=3,
                    keys=['image', 'label']),
                RandFlipd(prob=0.5, 
                    spatial_axis=2,
                    keys=['image', 'label']),
                RandFlipd(prob=0.5, 
                    spatial_axis=0,
                    keys=['image', 'label']),
                RandZoomd(prob=0.5, 
                    min_zoom=1.0,
                    max_zoom=1.2,
                    keep_size=True,
                    mode=['trilinear', 'nearest'],
                    keys=['image', 'label']),
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                # LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(
                    a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0,
                    keys=["image"]),
                SpatialCropd(
                    roi_start=[80,0,1],
                    roi_end=[240,320,61],
                    keys=["image", "label"]),
                ARGUS_RandSpatialCropSlicesd(
                    num_slices=num_slices,
                    axis=3,
                    center_slice=30,
                    keys=['image', 'label']),
                ToTensord(keys=["image", "label"]),
            ]
        )
        return {'train': train_transforms, 'val': val_transforms}


    def get_data(self):
        all_images = sorted(glob(os.path.join(self.hparams.data_dir, '*_?????.nii.gz')))
        all_labels = sorted(glob(os.path.join(self.hparams.data_dir, '*.extruded-overlay-NS.nii.gz')))
        return all_images, all_labels

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        
        img_paths, label_paths = self.get_data()
        if len(img_paths) < 1 or len(label_paths) < 1 or len(img_paths) != len(label_paths):
            log.error(f'Problem with image or label data in location {self.hparams.data_dir}. '  
                      f'Found Num images / labels = {len(img_paths)}/{len(label_paths)}')
        else:
            log.info(f"Num images / labels = {len(img_paths)}/{len(label_paths)}")

            
        self.subjects = []
        for image_path, label_path in zip(img_paths,label_paths):            
            # 'image' and 'label' are arbitrary names for the images
            # import ipdb
            # ipdb.set_trace()
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.subjects.append(subject)
        
        

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.hparams.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects

        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(
            dataset=self.subjects, 
            lengths=splits, 
            generator=torch.Generator().manual_seed(self.hparams.data_seed),)

        self.transforms = self.get_transforms()
        
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform['train'])
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.transform['test'])


    def train_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
