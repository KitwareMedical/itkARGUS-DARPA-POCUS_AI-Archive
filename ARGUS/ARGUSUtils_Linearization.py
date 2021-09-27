import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import mode

import skimage
from skimage import measure

import sys

import math

import itk
itkResampleImageUsingMapFilter = itk.itkARGUS.ResampleImageUsingMapFilter


####
# Estimate Zoom and Depth
####

def get_ruler_points_C52(im):
    """ Find points along ruler on left side of image """
    y_min = 80
    y_max = 1080
    min_x = 10
    max_x = 12
    mid = np.mean((im[y_min:y_max,min_x:max_x]//200),axis=1)
    y = np.flatnonzero(mid)
    return y

def get_depth_and_zoom_C52(im):
    y = get_ruler_points_C52(im)

    avg = 0
    count = 0
    yCenters = []
    for j in range(y.size-1):
        avg += y[j]
        count += 1
        if(y[j+1]-y[j]>5):
            avg /= count
            yCenters.append(avg)
            avg = 0
            count = 0
    avg += y[y.size-1]
    count += 1
    avg /= count
    yCenters.append(avg)
    avg = 0
    for j in range(len(yCenters)-1):
        avg += yCenters[j+1]-yCenters[j]
    avg /= len(yCenters)-1    
    
    tic_num = len(yCenters)
    tic_min = yCenters[0]
    tic_max = yCenters[len(yCenters)-1]
    tic_diff = avg
    if(tic_num==17):
        tic_depth = 16
        tic_scale = tic_diff/54.4375
    elif(tic_num==13):
        tic_depth = 12
        tic_scale = tic_diff/55.5
    elif(tic_num==11):
        tic_depth = 5
        tic_scale = tic_diff/38.0
    else:
        print("ERROR: Unknown image depth!")
        print("   Num tics = ", tic_num, "   diff = ", tic_diff)
        
    return tic_depth,tic_scale #,tic_num,tic_min,tic_max,tic_diff

def linearize_video( vid ):
    depth,zoom = get_depth_and_zoom_C52(vid[0])
    filename = "./linear_maps/linear_map_depth" + str(depth) + ".npy"
    mapping = np.load(filename)

    frame_size = np.shape(mapping)[:2]
    num_frames = vid.shape[0]
    vid_linear = np.empty((num_frames,frame_size[0],frame_size[1]))

    source_coords = mapping[:,:,:2].astype(int)
    source_coords_list = source_coords[:,:,::-1].flatten().tolist()
    kernels_list = mapping[:,:,2:].flatten().astype(float).tolist()

    ImageType = itk.Image[itk.F,2]
    linear_filter = itkResampleImageUsingMapFilter[ImageType,ImageType].New()
    linear_filter.SetOutputSize([frame_size[1],frame_size[0]])
    linear_filter.SetSourceMapping(source_coords_list)
    linear_filter.SetKernels(kernels_list)

    depth,zoom = get_depth_and_zoom_C52(vid[0])
    itkimgBase = itk.GetImageFromArray(vid.astype(np.float32))
    if(zoom!=1):
        print("Zoom factor =", zoom)
        itkimg = itk.GetImageFromArray(vid.astype(np.float32))
        itk_spacing = [1/zoom,1/zoom,1]
        itkimg.SetSpacing(itk_spacing)
        if(abs(zoom-1.26262627)<0.1):
            itk_origin = [192,108,0]
        elif(abs(zoom-1.012012012)<0.1):
            itk_origin = [0,0,0]
        elif(abs(zoom-0.804804805)):
            itk_origin = [-235,-170,0]
        itkimg.SetOrigin(itk_origin)
        resample = itk.ResampleImage.New(itkimg)
        resample.SetInterpolator("NearestNeighbor")
        resample.SetMatchImage(itkimgBase)
        resample.Update()
        vid = itk.GetArrayFromImage(resample.GetOutput())

    for i in range(num_frames):
        itkimg = itk.GetImageFromArray(vid[i].astype(np.float32))
        linear_filter.SetInput(itkimg)
        linear_filter.Update()
        vid_linear[i] = linear_filter.GetOutput()
    return vid_linear