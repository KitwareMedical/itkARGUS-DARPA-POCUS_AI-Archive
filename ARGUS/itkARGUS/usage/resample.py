import imageio
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import mode
import skimage
from skimage import measure
import sys

import itk

from time import perf_counter
from contextlib import contextmanager

itkResampleImageUsingMapFilter = itk.itkARGUS.ResampleImageUsingMapFilter

import site
site.addsitedir('../../')
from ARGUSUtils import *

@contextmanager
def timethis(name):
    start = perf_counter()
    yield
    print('Time for', name, 'is', perf_counter() - start)

def resample_speed_test(video, mapping):
    out_size = np.shape(mapping)[:2]

    src_coords = mapping[:,:,:2].astype(int)
    # tranpose, since the indexing is reversed
    #src_coords[:,:,[0,1]] = src_coords[:,:,[1,0]]
    source_maps = src_coords[:,:,::-1].flatten().tolist()

    kernels = mapping[:,:,2:].flatten().astype(float).tolist()

    def doit():
        res = np.ndarray((out_size[1],out_size[0],video.shape[2]))
        ImageType = itk.Image[itk.F, 2]
        F = itkResampleImageUsingMapFilter[ImageType, ImageType].New()
        F.SetOutputSize(out_size)
        F.SetSourceMapping(source_maps)
        F.SetKernels(kernels)
        with timethis('ResampleImageUsingMapFilter(30Frames):InnerLoop'):
            for i in range(video.shape[2]):
                itkimg = itk.GetImageFromArray(video[:,:,i])
                F.SetInput(itkimg)
                F.Update()
                res[:,:,i] = F.GetOutput()
        return res

    outimage = doit()
    #itk.imwrite(outimage, "output.nrrd")

vidFile = '../../../Data/TrainingData/AR-UNet/BAMC-PTXSliding/image_10391571128899_CLEAN.mp4'
vid = imageio.get_reader(vidFile, 'ffmpeg')

ims = vid.get_data(30)
ims = (ims / 255).astype(np.float32)
ocimg = np.mean(ims, axis=2)
mask = ocimg > 0.006
labelled = measure.label(mask)
center = np.asarray(np.shape(ocimg)) // 2
num = labelled[center[0], center[1]]
USmask = labelled == num

with timethis('GenerateImageMap'):
    mapping = get_rectilinear_resampling_map_corners(USmask, ray_density=2/3, blur=0.4)

with timethis('ResampleImageUsingMapFilter(30Frames)'):
    result = resample_speed_test(ims, mapping)
