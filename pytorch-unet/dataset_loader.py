import random
import numpy as np
import scipy as sp
from scipy import ndimage
import torch
import torchvision.transforms as T

class DataAugmentor(torch.utils.data.Dataset):
    """
    Data generator (PyTorch term is 'Dataset') which stores the data
    and handles augmentations
    """
    def __init__(self,
                 x_set, 
                 y_set,
                 image_dimensions,
                 max_rotation_angle = 10.0,
                 max_shift_factor=0.1,
                 min_zoom_factor=0.9,
                 max_zoom_factor=1.1,
                 n_classes=2):
        self.x = x_set
        self.y = y_set
        self.image_dimensions = image_dimensions
        self.n_classes = n_classes
        self.number_of_images = self.x.shape[0]
        self.indexes = np.arange(self.number_of_images)
        self.transform = T.RandomAffine(degrees=max_rotation_angle,
                                        translate = (max_shift_factor, max_shift_factor),
                                        scale = (min_zoom_factor, max_zoom_factor))
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return int(self.number_of_images)

    def __getitem__(self, index):
        """
        Get single (not batched) image and target for that image
        In PyTorch, targets are NOT one-hot
        In PyTorch, channels-first is the convention
        Assume that the data is greyscale (like ultrasound)
        """

        x = torch.empty((1, *self.image_dimensions))
        y = torch.empty((1, *self.image_dimensions))

        flip_flag = np.random.randint(2)
        if flip_flag == 1:
            x = torch.flip(self.x[index], dims=(2,))
            y = torch.flip(self.y[index], dims=(2,))
        else:
            x = self.x[index]
            y = self.y[index]

        x_out = self.transform(x)
        y_out = self.transform(y)

        x_out = torch.clip(x_out, 0.0, 1.0)
        y_out = torch.clip(y_out, 0.0, 1.0)

        return x_out, y_out
