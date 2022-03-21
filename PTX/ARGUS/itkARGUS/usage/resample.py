import imageio
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import mode
import skimage
from skimage import measure
import sys

import itk

import site
site.addsitedir('../../')
from ARGUSUtils import *
from ARGUSUtils_IO import *

itkResampleImageUsingMapFilter = itk.itkARGUS.ResampleImageUsingMapFilter

def resample_speed_test(video, mapping):
    out_size = np.shape(mapping)[:2]
    itk_out_size = [out_size[1],out_size[0]]

    src_coords = mapping[:,:,:2].astype(int)
    # tranpose, since the indexing is reversed
    #src_coords[:,:,[0,1]] = src_coords[:,:,[1,0]]
    source_maps = src_coords[:,:,::-1].flatten().tolist()

    kernels = mapping[:,:,2:].flatten().astype(float).tolist()

    def doit():
        res = np.ndarray((video.shape[0],out_size[0],out_size[1]))
        ImageType = itk.Image[itk.F, 2]
        F = itkResampleImageUsingMapFilter[ImageType, ImageType].New()
        F.SetOutputSize(itk_out_size)
        F.SetSourceMapping(source_maps)
        F.SetKernels(kernels)
        with time_this('ResampleImageUsingMapFilter(30Frames):InnerLoop'):
            for i in range(video.shape[0]):
                itkimg = itk.GetImageFromArray(video[i,:,:].astype(np.float32))
                F.SetInput(itkimg)
                F.Update()
                res[i,:,:] = F.GetOutput()
        return res

    outimage = doit()
    return outimage

vidFile = '../../../Data/TrainingData/AR-UNet/BAMC-PTXSliding/image_10391571128899_CLEAN.mp4'
ims = load_video(vidFile)

mapping = np.load("../../linear_maps/linear_map_depth12.npy")

with time_this('ResampleImageUsingMapFilter(30Frames)'):
    result = resample_speed_test(ims, mapping)

itk.imwrite(itk.GetImageFromArray(result.astype(np.float32)), "usage_output.nrrd")
