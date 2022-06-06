import os
from glob import glob
from typing import Optional
import torch

from pocusnet.vendor.argus_transforms import ARGUS_RandSpatialCropSlicesd
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (AsChannelFirstd, Compose,
                              LoadImaged, Resized,
                              ScaleIntensityRanged,
                              ToTensord, RandSpatialCropd)
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split

from pocusnet import utils

log = utils.get_logger(__name__)


class PNBDataModule(LightningDataModule):
    """Example of LightningDataModule for PNB data.

    For more information, read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

    To test:
        >>> import pocusnet.datamodules.pnb_datamodule as pnb_d
        >>> d = pnb_d.PNBDataModule(data_dir="/data/krsdata2-pocus-ai-synced/root/Data_PNB/Annotations/annptations_yuri/")
        >>> d.prepare_data()
        >>> d.setup()
    """

    def __init__(
        self,
        data_dir: str = "data/PNB",
        train_val_ratio: float = 0.8,
        data_seed: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 2,
        num_slices: int = 32,
        size_x: int = 320,
        size_y: int = 640
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.subjects = None

        # data transformations
        self.transforms = self.get_transforms

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def get_transforms(self):
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
                    num_slices=[self.hparams.num_slices, 1],
                    axis=0,
                    reduce_to_statistics=[True, False],
                    extended=False,
                    keys=['image', 'label']),
                Resized(
                    spatial_size=(-1, 640),
                    mode=["bilinear", "nearest"],
                    keys=['image', 'label']
                ),
                RandSpatialCropd(
                    roi_size=(self.hparams.size_x, self.hparams.size_y),
                    random_center=True,
                    random_size=False,
                    keys=['image', 'label']
                ),
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
                    num_slices=[self.hparams.num_slices, 1],
                    center_slice=self.hparams.num_slices / 2,
                    axis=0,
                    reduce_to_statistics=[True, False],
                    extended=False,
                    keys=['image', 'label']),
                Resized(
                    spatial_size=(-1, 640),
                    mode=["bilinear", "nearest"],
                    keys=['image', 'label']
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
        return {'train': train_transforms, 'test': val_transforms}

    def get_data(self):
        all_images = sorted(
            glob(
                os.path.join(
                    self.hparams.data_dir,
                    '*_cropM.nii.gz')))
        all_labels = sorted(
            glob(
                os.path.join(
                    self.hparams.data_dir,
                    '*.overlay.mha')))
        return all_images, all_labels

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """

        img_paths, label_paths = self.get_data()
        if len(img_paths) < 1 or len(label_paths) < 1 or len(
                img_paths) != len(label_paths):
            log.error(f'Problem with image or label data in location {self.hparams.data_dir}. '
                      f'Found Num images / labels = {len(img_paths)}/{len(label_paths)}')
        else:
            log.info(
                f"Num images / labels = {len(img_paths)}/{len(label_paths)}")

        self.subjects = []
        for image_path, label_path in zip(img_paths, label_paths):
            subject = {'image': image_path, 'label': label_path}
            self.subjects.append(subject)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        num_subjects = len(self.subjects)
        num_train_subjects = int(
            round(
                num_subjects *
                self.hparams.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects

        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(
            dataset=self.subjects,
            lengths=splits,
            generator=torch.Generator().manual_seed(self.hparams.data_seed),)

        self.transform = self.get_transforms()

        log.info(f'Loading Train Dataset {len(train_subjects)}/{len(self.subjects)}')
        self.train_set = CacheDataset(
            train_subjects,
            transform=self.transform['train'],
            num_workers=self.hparams.num_workers)

        log.info(f'Loading Validation Dataset {len(val_subjects)}/{len(self.subjects)}')
        self.val_set = CacheDataset(
            val_subjects,
            transform=self.transform['test'],
            num_workers=self.hparams.num_workers)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
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
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


def demo():
    import matplotlib.pyplot as plt
    import pocusnet.datamodules.pnb_datamodule as pnb_d
    d = pnb_d.PNBDataModule(
        data_dir="/data/krsdata2-pocus-ai-synced/root/Data_PNB/Annotations/annotations_yuri/")
    d.prepare_data()
    d.setup()
    train_dataloader = d.train_dataloader()

    from monai.utils import first
    check_data = first(train_dataloader)

    val_dataloader = d.val_dataloader()
    check_data = first(val_dataloader)
    imgnum = 1
    image, label = (check_data["image"][imgnum],
                    check_data["label"][imgnum])
    print(check_data["image"].shape)
    print(image.shape)
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    plt.figure("check", (12, 6))
    plt.subplot(1, 4, 1)
    plt.title("image mean")
    plt.imshow(image[0, :, :])
    plt.subplot(1, 4, 2)
    plt.imshow(image[1, :, :])
    plt.title("image std")
    plt.subplot(1, 4, 3)
    plt.imshow(label[0, :, :])
    plt.title("label")
    plt.show()
    plt.savefig("dummy_name.png")
    print(f'Label value min:{label.min()} and max {label.max()}')

if __name__ == "__main__":
    demo()