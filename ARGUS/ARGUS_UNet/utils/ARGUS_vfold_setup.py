import os
import numpy as np
from glob import glob
from monai.data import CacheDataset, DataLoader
import ubelt as ub
import pint
Ureg = pint.UnitRegistry()

def setup_vfold_files(self, img_dir, anno_dir, p_prefix, n_prefix):
    self.all_train_images = sorted(glob(os.path.join(img_dir, self.img_file_extension)))
    self.all_train_labels = sorted(glob(os.path.join(anno_dir, self.label_file_extension)))

    total_bytes = 0
    for p in self.all_train_images:
        p = ub.Path(p)
        total_bytes += p.stat().st_size
    print("\n")
    print("Total size of images in the dataset: ")
    print((total_bytes * Ureg.byte).to("GiB"))

    total_bytes = 0
    for p in self.all_train_labels:
        p = ub.Path(p)
        total_bytes += p.stat().st_size
    print("\n")
    print("Total size of labels in the dataset: ")    
    print((total_bytes * Ureg.byte).to("GiB"))

    num_images = len(self.all_train_images)
    
    print("\n")
    print("Num images / labels =", num_images, len(self.all_train_labels))
    print("\n")

    fold_prefix_list = []
    p_count = 0
    n_count = 0
    for i in range(self.num_folds):
        num_p = 1
        num_n = 1
        if i > self.num_folds - 2:
            if i % 2 == 0:
                num_p = 2
                num_n = 1
            else:
                num_p = 1
                num_n = 1
        f = []
        if p_count < len(p_prefix):
            for p in range(num_p):
                f.append([p_prefix[p_count + p]])
        p_count += num_p
        if n_count < len(n_prefix):
            for n in range(num_n):
                f.append([n_prefix[n_count + n]])
        n_count += num_n
        fold_prefix_list.append(f)

    for i in range(self.num_folds):
        print(i, fold_prefix_list[i])
    print("\n")

    self.train_files = []
    self.val_files = []
    self.test_files = []

    for i in range(self.num_folds):
        tr_folds = []
        va_folds = []
        for f in range(i, i + self.num_folds - 3):
            tr_folds.append(fold_prefix_list[f % self.num_folds])
        tr_folds = list(np.concatenate(tr_folds).flat)
        for f in range(i + self.num_folds - 3, i + self.num_folds - 1):
            va_folds.append(fold_prefix_list[f % self.num_folds])
        va_folds = list(np.concatenate(va_folds).flat)
        te_folds = list(
            np.concatenate(
                fold_prefix_list[(i + self.num_folds - 1) % self.num_folds]
            ).flat
        )
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
        print(
            len(self.train_files[i]),
            len(self.val_files[i]),
            len(self.test_files[i]),
        )

def setup_training_vfold(self, vfold_num):
    self.vfold_num = vfold_num

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
    )

    val_ds = CacheDataset(
        data=self.val_files[self.vfold_num],
        transform=self.val_transforms,
        cache_rate=self.cache_rate_val,
        num_workers=self.num_workers_val,
    )
    self.val_loader = DataLoader(
        val_ds, batch_size=self.batch_size_val, num_workers=self.num_workers_val
    )

def setup_testing_vfold(self, vfold_num):
    self.vfold_num = vfold_num

    test_ds = CacheDataset(
        data=self.test_files[self.vfold_num],
        transform=self.test_transforms,
        cache_rate=self.cache_rate_test,
        num_workers=self.num_workers_test,
    )
    self.test_loader = DataLoader(
        test_ds, batch_size=self.batch_size_test, num_workers=self.num_workers_test
    )