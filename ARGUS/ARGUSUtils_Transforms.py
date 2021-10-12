import numpy as np

from monai.transforms.transform import Randomizable, Transform, MapTransform, RandomizableTransform
from monai.transforms.inverse import InvertibleTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import IndexSelection, KeysCollection

from monai.transforms import (
    SpatialCrop,
)

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

class ARGUS_RandSpatialCropSlices(RandomizableTransform, Transform):
    """
    ARGUS specific cropping class to extract adjacent slices along an axis.
    Slice-based cropper to extract random adjacent slices from a volume.
    It supports cropping ND spatial (channel-first) data.
    """
    backend = SpatialCrop.backend

    def __init__(
        self,
        num_slices: int = 21,
        axis: int = -1,
        center_slice: int = -1,
    ) -> None:
        RandomizableTransform.__init__(self, 1.0)
        self.num_slices = num_slices
        self.axis = axis
        self.center_slice = center_slice
        self._roi_start: Optional[Sequence[int]] = None
        self._roi_center_slice: int = -1
        self._roi_end: Optional[Sequence[int]] = None
    
    def randomize(self, data: NdarrayOrTensor) -> None:
        if self.center_slice == -1:
            boundary = self.num_slices//2
            buffer = 0
            # if even num_slices, add a buffer so last slice can be used
            if boundary*2 == self.num_slices:
                buffer = 1
            self._roi_center_slice = self.R.randint(boundary, data.shape[self.axis]-boundary+buffer)
        else:
            self._roi_center_slice = self.center_slice

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        self.randomize(data=img[0])

        orig_size = img.shape[1:]
        self._roi_start = np.zeros((len(orig_size)), dtype=np.int32)
        
        boundary = self.num_slices//2
        tlist = list(self._roi_start)
        tlist[self.axis] = self._roi_center_slice - boundary
        self._roi_start = tuple(tlist)
        
        self._roi_end = orig_size
        tlist = list(self._roi_end)
        tlist[self.axis] = self._roi_start[self.axis] + self.num_slices
        self._roi_end = tuple(tlist)

        cropper = SpatialCrop(roi_start=self._roi_start, roi_end=self._roi_end)

        return cropper(img)

class ARGUS_RandSpatialCropSlicesd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`ARGUS_RandSpatialCropSlices`.
    Slice-based cropper to extract random adjacent slices from a volume.
    It supports cropping ND spatial (channel-first) data.
    """

    backend = ARGUS_RandSpatialCropSlices.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_slices: int = 21,
        axis: int = -1,
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
        #super().__init__(keys, allow_missing_keys)
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self)
        self.axis = axis
        self.num_slices = num_slices
        self.center_slice = center_slice
        self._roi_start: Optional[Sequence[int]] = None
        self._roi_center_slice: int = -1
        self._roi_end: Optional[Sequence[int]] = None

    def randomize(self, data: NdarrayOrTensor) -> None:
        super().randomize(data)
        if self.center_slice == -1:
            boundary = self.num_slices//2
            buffer = 0
            # if even num_slices, add a buffer so last slice can be used
            if boundary*2 == self.num_slices:
                buffer = 1
            self._roi_center_slice = self.R.randint(boundary, data.shape[self.axis]-boundary+buffer)
        else:
            self._roi_center_slice = self.center_slice

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        self.randomize(data=data[self.keys[0]][0])
        d = dict(data)

        cropper = ARGUS_RandSpatialCropSlices(num_slices=self.num_slices, axis=self.axis, center_slice=self._roi_center_slice)
        for key in self.key_iterator(d):
            orig_size = d[key].shape[1:]
            d[key] = cropper(d[key])
            self.push_transform(
                d,
                key,
                orig_size=orig_size,
                extra_info={
                    "num_slices": self.num_slices,
                    "center_slice": self._roi_center_slice,
                    "axis": self.axis})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)

            orig_size = transform[InverseKeys.ORIG_SIZE]
            self.num_slices = transform[InverseKeys.NUM_SLICES]
            self._roi_center_slice = transform[InverseKeys.CENTER_SLICE]
            self.axis = transform[InverseKeys.AXIS]

            pad_to_start = np.empty((len(orig_size)), dtype=np.int32)
            pad_to_end = np.empty((len(orig_size)), dtype=np.int32)
            
            boundary = self.num_slices//2
        
            self._roi_start = np.zeros((len(orig_size)), dtype=np.int32)
            tlist = list(self._roi_start)
            tlist[self.axis] = self._roi_center_slice - boundary
            self._roi_start = tuple(tlist)

            self._roi_end = orig_size
            tlist = list(self._roi_end)
            tlist[self.axis] = self._roi_start[self.axis] + self.num_slices
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
