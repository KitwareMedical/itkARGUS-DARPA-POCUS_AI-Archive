import numpy as np
import scipy as sp


#from skimage import data
#from skimage.util import img_as_float
#from skimage.filters import gabor_kernel

from monai.transforms import (
    SpatialCrop
)
from monai.transforms.transform import (
    #Randomizable,
    Transform,
    MapTransform,
    RandomizableTransform
)
from monai.transforms.inverse import (
    InvertibleTransform
)
from monai.config import (
    #IndexSelection,
    KeysCollection
)
from monai.config.type_definitions import (
    NdarrayOrTensor
)
from monai.utils import (
    ensure_tuple_rep
)
#from monai.utils.type_conversion import (
    #convert_data_type,
    #convert_to_dst_type
#)

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
        num_slices: int = 16,
        axis: int = -1,
        center_slice: int = 99999,
        boundary: int = -1,
        require_labeled: bool = False,
        reduce_to_statistics: bool = False,
        extended: bool = False,
        include_center_slice: bool = False,
        include_mean_abs_diff: bool = False,
        include_skewness: bool = False,
        include_kurtosis: bool = False,
        include_gradient: bool = False
    ) -> None:
        RandomizableTransform.__init__(self, 1.0)
        self.num_slices = num_slices
        self.axis = axis
        self.center_slice = center_slice
        self.boundary = boundary
        self.require_labeled = require_labeled
        self.reduce_to_statistics = reduce_to_statistics
        self.extended = extended
        self.include_center_slice = include_center_slice
        self.include_mean_abs_diff = include_mean_abs_diff
        self.include_skewness = include_skewness
        self.include_kurtosis = include_kurtosis
        self.include_gradient = include_gradient
        self._roi_start: Optional[Sequence[int]] = None
        self._roi_center_slice: int = 99999
        self._roi_end: Optional[Sequence[int]] = None
        if self.boundary == -1:
            self.boundary = self.num_slices//2

    
    def randomize(self, data: NdarrayOrTensor) -> None:
        if self.center_slice == 99999:
            buffer = 0
            # if even num_slices, add a buffer so last slice can be used
            if self.boundary*2 == self.num_slices:
                buffer = 1
            while True:
                self._roi_center_slice = self.R.randint(self.boundary,
                        data.shape[self.axis]-self.boundary+buffer)
                if not self.require_labeled:
                    break
                else:
                    slice_roi = [slice(0,data.shape[i])
                                 for i in range(len(data.shape))]
                    slice_roi[self.axis] = self._roi_center_slice
                    is_labeled = (data[tuple(slice_roi)]!=0).any()
                    if is_labeled:
                        break
        else:
            self._roi_center_slice = self.center_slice

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing DOES apply to the channel dim.
        """
        self.randomize(img)

        def make_slices(smin, smax, _start, _end):
            tlist = list(_start)
            tlist[self.axis] = smin
            new_start = tuple(tlist)
        
            tlist = list(_end)
            tlist[self.axis] = smax
            new_end = tuple(tlist)

            slices = [slice(int(s), int(e))
                       for s, e in zip(new_start, new_end)]
            return new_start, new_end, slices

        _size = img.shape
        _start = np.zeros((len(_size)), dtype=np.int32)
        _end = _size

        if self._roi_center_slice < 0:
            self._roi_center_slice = ( 
                img.shape[self.axis] + self._roi_center_slice )
        
        self._roi_start, self._roi_end, slices_all = make_slices(
                self._roi_center_slice - self.boundary,
                self._roi_center_slice - self.boundary + self.num_slices,
                _start, _end )

        arr = img[tuple(slices_all)]

        if not self.reduce_to_statistics:
            return arr

        num_stats = 2
        if self.include_center_slice:
            num_stats += 1
        if self.include_mean_abs_diff:
            num_stats += 1
        if self.include_skewness:
            num_stats += 1
        if self.include_kurtosis:
            num_stats += 1
        if self.include_gradient:
            num_stats += len(img.shape)
        if self.extended:
            num_stats *= 2

        out_img_shape = list(img.shape)
        if self.axis != 0:
            out_img_shape[:self.axis] = out_img_shape[1:self.axis+1]
        out_img_shape[0] = num_stats
        out_img = np.empty(tuple(out_img_shape))

        out_img_slice = 0
        if not self.extended:
            np.mean(arr,axis=self.axis,out=out_img[out_img_slice])
            out_img_slice += 1
            np.std(arr,axis=self.axis,out=out_img[out_img_slice])
            out_img_slice += 1

            if self.include_center_slice:
                r_min = self.num_slices // 2
                r_max = r_min + 1
                r_min,r_max,tmp_slices = make_slices(r_min,r_max,
                                                     _start,_end)
                out_img[out_img_slice] = arr[tuple(tmp_slices)]
                out_img_slice += 1
            if self.include_mean_abs_diff:
                np.mean(np.absolute(np.diff(arr,axis=self.axis)),
                        axis=self.axis,
                        out=out_img[out_img_slice])
                out_img_slice += 1
            if self.include_skewness:
                out_img[out_img_slice] = sp.stats.skew(arr,
                                                       axis=self.axis)
                out_img_slice += 1
            if self.include_kurtosis:
                out_img[out_img_slice] = sp.stats.kurtosis(arr,
                                                           axis=self.axis)
                out_img_slice += 1
            if self.include_gradient:
                gradlist = np.gradient(arr)
                for d in range(len(gradlist)):
                    np.mean(gradlist[d],axis=self.axis,
                        out=out_img[out_img_slice])
                    out_img_slice += 1
        else:
            ext_img_shape = out_img_shape
            ext_img_shape[0] = 3
            ext_img = np.empty(tuple(ext_img_shape))

            ext_unit = self.num_slices / 7
            ext_num_slices = 3 * ext_unit
            _start = np.zeros((len(arr.shape)), dtype=np.int32)
            _end = arr.shape

            r_min = np.zeros((3))

            r_min[0] = 0
            r_max = r_min[0] + ext_num_slices
            _, _, r_slices = make_slices(r_min[0],r_max,_start,_end)
            slices = [r_slices]

            r_min[1] = 2*ext_unit
            r_max = r_min[1] + ext_num_slices
            _, _, r_slices = make_slices(r_min[1],r_max,_start,_end)
            slices.append(r_slices)

            r_min[2] = 4*ext_unit
            r_max = r_min[2] + ext_num_slices
            _, _, r_slices = make_slices(r_min[2],r_max,_start,_end)
            slices.append(r_slices)

            # Mean
            for i in range(len(slices)):
                np.mean(arr[tuple(slices[i])],axis=self.axis,out=ext_img[i])
            np.amax(ext_img,axis=0,out=out_img[out_img_slice])
            out_img_slice += 1
            np.amin(ext_img,axis=0,out=out_img[out_img_slice])
            out_img_slice += 1

            # Std Dev
            for i in range(len(slices)):
                np.std(arr[tuple(slices[i])],axis=self.axis,out=ext_img[i])
            np.amax(ext_img,axis=0,out=out_img[out_img_slice])
            out_img_slice += 1
            np.amin(ext_img,axis=0,out=out_img[out_img_slice])
            out_img_slice += 1

            if self.include_center_slice:
                for i in range(len(slices)):
                    r_max = r_min[i] + 1
                    _,_,tmp_slices = make_slices(r_min[i],r_max,_start,_end)
                    ext_img[i] = arr[tuple(tmp_slices)]
                np.amax(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1
                np.amin(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1

            if self.include_mean_abs_diff:
                tmp_arr = np.absolute(np.diff(arr,axis=self.axis))
                for i in range(len(slices)):
                    np.mean(tmp_arr[tuple(slices[i])],
                            axis=self.axis,
                            out=ext_img[i])
                np.amax(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1
                np.amin(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1

            if self.include_skewness:
                for i in range(len(slices)):
                    ext_img[i] = sp.stats.skew(arr[tuple(slices[i])],
                                               axis=self.axis)
                np.amax(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1
                np.amin(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1

            if self.include_kurtosis:
                for i in range(len(slices)):
                    ext_img[i] = sp.stats.kurtosis(arr[tuple(slices[i])],
                                               axis=self.axis)
                np.amax(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1
                np.amin(ext_img,axis=0,out=out_img[out_img_slice])
                out_img_slice += 1

            if self.include_gradient:
                gradlist = np.gradient(arr)
                for d in range(len(gradlist)):
                    for i in range(len(slices)):
                        np.mean(gradlist[d][tuple(slices[i])],
                                axis=self.axis,
                                out=ext_img[i])
                    np.amax(ext_img,axis=0,out=out_img[out_img_slice])
                    out_img_slice += 1
                    np.amin(ext_img,axis=0,out=out_img[out_img_slice])
                    out_img_slice += 1
            

        return out_img



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
        num_slices: Union[Sequence[int], int] = 16,
        axis: int = -1,
        center_slice: int = 99999,
        boundary: int = -1,
        require_labeled: bool = False,
        reduce_to_statistics: Union[Sequence[bool], bool] = False,
        extended: bool = False,
        include_center_slice: bool = False,
        include_mean_abs_diff: bool = False,
        include_skewness: bool = False,
        include_kurtosis: bool = False,
        include_gradient: bool = False,
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
            require_labeled: using the last key's data, only select slices that aren't blank
            include_center_slice: If enabled, the raw center frame from the chosen window of frames is concatenated with the Mean/Std statistics.
            include_mean_abs_diff: If enabled, the mean absolute value of the difference between adjacent slices is concatenated with the Mean/Std statistics.
            include_skewness: If enabled, the skewness across the slices is concatenated with the Mean/Std statistics.
            include_kurtosis: If enabled, the kurtosis across the slices is concatenated with the Mean/Std statistics.
            include_gradient: If enabled, the gradient in each dimension is concatenated with the Mean/Std statistics.
            allow_missing_keys: don't raise exception if key is missing.
        """
        #super().__init__(keys, allow_missing_keys)
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self)
        self.axis = axis
        self.num_slices = ensure_tuple_rep(num_slices, len(self.keys))
        self.boundary = boundary
        self.require_labeled = require_labeled
        self.center_slice = center_slice
        self.reduce_to_statistics = ensure_tuple_rep(reduce_to_statistics, len(self.keys))
        self.extended = extended
        self.include_center_slice = include_center_slice
        self.include_mean_abs_diff = include_mean_abs_diff
        self.include_skewness = include_skewness
        self.include_kurtosis = include_kurtosis
        self.include_gradient = include_gradient
        self._roi_start: Optional[Sequence[int]] = None
        self._roi_center_slice: int = 99999
        self._roi_end: Optional[Sequence[int]] = None
        if self.boundary == -1:
            self.boundary = max(self.num_slices)//2

    def randomize(self, data: NdarrayOrTensor) -> None:
        super().randomize(data)
        if self.center_slice == 99999:
            buffer = 0
            # if even num_slices, add a buffer so last slice can be used
            if self.boundary*2 == max(self.num_slices):
                buffer = 1
            while True:
                self._roi_center_slice = self.R.randint(self.boundary,
                        data.shape[self.axis]-self.boundary+buffer)
                if not self.require_labeled:
                    break
                else:
                    slice_roi = [slice(0,data.shape[i])
                            for i in range(len(data.shape))]
                    slice_roi[self.axis] = self._roi_center_slice
                    is_labeled = (data[tuple(slice_roi)]!=0).any()
                    if is_labeled:
                        break
        else:
            self._roi_center_slice = self.center_slice

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        ## passing last key's data to randomize, for require_labeled
        self.randomize(data=data[self.keys[-1]])
        d = dict(data)

        for key, num_slices, reduce_to_statistics in self.key_iterator(d, self.num_slices, self.reduce_to_statistics):
            cropper = ARGUS_RandSpatialCropSlices(num_slices=num_slices, axis=self.axis, center_slice=self._roi_center_slice, reduce_to_statistics=reduce_to_statistics,boundary=self.boundary,require_labeled=self.require_labeled,extended=self.extended,include_center_slice=self.include_center_slice,include_mean_abs_diff=self.include_mean_abs_diff,include_skewness=self.include_skewness,include_kurtosis=self.include_kurtosis,include_gradient=self.include_gradient)
            orig_size = d[key].shape
            d[key] = cropper(d[key])
            self.push_transform(
                d,
                key,
                orig_size=orig_size,
                extra_info={
                    "num_slices": num_slices,
                    "center_slice": self._roi_center_slice,
                    "axis": self.axis})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)

            orig_size = transform[InverseKeys.ORIG_SIZE]
            num_slices = transform[InverseKeys.EXTRA_INFO]["num_slices"]
            self._roi_center_slice = transform[InverseKeys.EXTRA_INFO]["center_slice"]
            self.axis = transform[InverseKeys.EXTRA_INFO]["axis"]

            pad_to_start = np.empty((len(orig_size)), dtype=np.int32)
            pad_to_end = np.empty((len(orig_size)), dtype=np.int32)
            
            boundary = num_slices//2
        
            self._roi_start = np.zeros((len(orig_size)), dtype=np.int32)
            tlist = list(self._roi_start)
            tlist[self.axis] = self._roi_center_slice - boundary
            self._roi_start = tuple(tlist)

            self._roi_end = orig_size
            tlist = list(self._roi_end)
            tlist[self.axis] = self._roi_start[self.axis] + num_slices
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
