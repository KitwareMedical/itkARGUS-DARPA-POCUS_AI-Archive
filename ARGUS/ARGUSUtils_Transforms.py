import numpy as np

from monai.transforms.transform import Randomizable, Transform, MapTransform
from monai.transforms.inverse import InvertibleTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import IndexSelection, KeysCollection

from monai.transforms import (
    SpatialCrop,
)

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

class ARGUS_RandSpatialCropSlices(Randomizable, Transform):
    backend = SpatialCrop.backend

    def __init__(
        self,
        num_slices: int = 21,
        center_slice: int = -1,
    ) -> None:
        self.num_slices = num_slices
        self.center_slice = center_slice
        self._roi_start: Optional[Sequence[int]] = None
        self._roi_end: Optional[Sequence[int]] = None
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        orig_size = img.shape[1:]
        boundary = self.num_slices//2
        
        self._roi_start = np.zeros((len(orig_size)), dtype=np.int32)
        buffer = 0
        if boundary*2 == self.num_slices:  # if even num_slices, add a buffer so last slice can be used
            buffer = 1
        
        if self.center_slice == -1:
            self.center_slice =  self.R.randint(boundary, img.shape[-1]-boundary+buffer)
        tlist = list(self._roi_start)
        tlist[-1] = self.center_slice - boundary
        self._roi_start = tuple(tlist)
        
        self._roi_end = orig_size
        tlist = list(self._roi_end)
        tlist[-1] = self._roi_start[-1] + self.num_slices
        self._roi_end = tuple(tlist)
        cropper = SpatialCrop(roi_start=self._roi_start, roi_end=self._roi_end)
        return cropper(img)

class ARGUS_RandSpatialCropSlicesd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialCrop`.
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    backend = ARGUS_RandSpatialCropSlices.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_slices: int = 21,
        center_slice: int = -1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.cropper = ARGUS_RandSpatialCropSlices(num_slices)
        self.num_slices = num_slices
        self.center_slice = center_slice
        self._roi_start: Optional[Sequence[int]] = None
        self._roi_end: Optional[Sequence[int]] = None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            orig_size = d[key].shape[1:]
            d[key] = self.cropper(d[key])
            self.push_transform(d, key, orig_size=orig_size)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform[InverseKeys.ORIG_SIZE]
            pad_to_start = np.empty((len(orig_size)), dtype=np.int32)
            pad_to_end = np.empty((len(orig_size)), dtype=np.int32)
            
            boundary = self.num_slices//2
        
            self._roi_start = np.zeros((len(orig_size)), dtype=np.int32)
            buffer = 0
            if boundary*2 == self.num_slices:  # if even num_slices, add a buffer so last slice can be used
                buffer = 1
            if self.center_slice == -1:
                self.center_slice =  self.R.randint(boundary, img.shape[-1]-boundary+buffer)
            tlist = list(self._roi_start)
            tlist[-1] = self.center_slice - boundary
            self._roi_start = tuple(tlist)

            self._roi_end = orig_size
            tlist = list(self._roi_end)
            tlist[-1] = self._roi_start[-1] + self.num_slices
            self._roi_end = tuple(tlist)
        
            pad_to_start = self._roi_start
            pad_to_end = self._roi_end
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
            
        return d

