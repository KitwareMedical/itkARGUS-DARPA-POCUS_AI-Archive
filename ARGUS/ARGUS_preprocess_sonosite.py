import numpy as np
from os import path

import itk
itkResampleImageUsingMapFilter = itk.itkARGUS.ResampleImageUsingMapFilter

####
# Estimate Zoom and Depth
####

class ARGUS_preprocess_sonosite():
    
    def __init__(self, new_size=None):
        self.new_size = new_size
        
    def get_ruler_points(self, im):
        """ Find points along ruler on left side of image """
        y_min = 80
        y_max = 1080
        min_x = 10
        max_x = 12
        mid = np.mean((im[y_min:y_max,min_x:max_x]//200),axis=1)
        y = np.flatnonzero(mid)+80
        return y
    
    def get_depth_and_zoom(self, im):
        y = self.get_ruler_points(im)

        assert len(y) > 5, "Could not find ruler in Sonosite format."
        assert len(y) < 75, "Could not find ruler in Sonosite format."
    
        im_shape = im.shape
    
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

        assert len(yCenters) > 5, "Could not find ruler in Sonosite format."
        assert len(yCenters) < 75, "Could not find ruler in Sonosite format."

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
            tic_offsetY = 181.0-tic_min/tic_scale
            tic_offsetX = tic_offsetY * (im_shape[1]/im_shape[0])
        elif(tic_num==13):
            tic_depth = 12
            tic_scale = tic_diff/55.5
            tic_offsetY = 258.5-tic_min/tic_scale
            tic_offsetX = tic_offsetY * (im_shape[1]/im_shape[0])
        elif(tic_num==11):
            tic_depth = 5
            tic_scale = tic_diff/38.0
            tic_offsetY = 436.0-tic_min/tic_scale
            tic_offsetX = tic_offsetY * (im_shape[1]/im_shape[0])
        else:
            print("ERROR: Unknown image depth!")
            print("   Num tics = ", tic_num, "   diff = ", tic_diff)
            
        return tic_depth,tic_scale,tic_offsetX,tic_offsetY #,tic_num,tic_min,tic_max,tic_diff
    
    def unzoom_video(self, vid, zoom, offsetY):
        if(zoom!=1):
            itkimgBase = itk.GetImageViewFromArray(vid.astype(np.float32))
    
            itkimg = itk.GetImageViewFromArray(vid.astype(np.float32))
            itk_spacing = [1/zoom,1/zoom,1]
            itkimg.SetSpacing(itk_spacing)
            if(abs(zoom-1.26262627)<0.1):
                itk_origin = [200,offsetY,0] #192,offsetY,0
            elif(abs(zoom-1.012012012)<0.1):
                itk_origin = [7,offsetY,0]
            elif(abs(zoom-0.804804805)<0.1):
                itk_origin = [-235,offsetY,0]
            else:
                print( f"ERROR: zoom factor {zoom} not known." )
                return vid
            itkimg.SetOrigin(itk_origin)
            
            resample = itk.ResampleImage.New(itkimg)
            resample.SetInterpolator("NearestNeighbor")
            resample.SetMatchImage(itkimgBase)
            resample.Update()
            vid_res = itk.GetArrayFromImage(resample.GetOutput())
            return vid_res
        else:
            return vid
    
    def process(self, vid_img):
        vid = itk.GetArrayViewFromImage(vid_img)
        
        depth,zoom,offsetX,offsetY = self.get_depth_and_zoom(vid[0])
        filename = path.join(path.dirname(__file__), 'linearization_maps_sonosite', f'linear_map_depth{str(depth)}.npy')
        mapping = np.load(filename)
    
        frame_size = np.shape(mapping)[:2]
        num_frames = vid.shape[0]
        vid_linear = np.empty((num_frames,frame_size[0],frame_size[1]))
    
        source_coords = mapping[:,:,:2].astype(int)
        source_coords_list = source_coords[:,:,::-1].flatten().tolist()
        kernels_list = mapping[:,:,2:].flatten().astype(float).tolist()
    
        vid = self.unzoom_video(vid, zoom, offsetY)
    
        spacing2D = [1, 1]
        ImageType = itk.Image[itk.F,2]
        linear_filter = itkResampleImageUsingMapFilter[ImageType,ImageType].New()
        linear_filter.SetOutputSize([frame_size[1],frame_size[0]])
        linear_filter.SetSourceMapping(source_coords_list)
        linear_filter.SetKernels(kernels_list)
        for i in range(num_frames):
            itkimg = itk.GetImageFromArray(vid[i].astype(np.float32))
            linear_filter.SetInput(itkimg)
            linear_filter.Update()
            vid_linear[i] = linear_filter.GetOutput()
            if i == 0:
                spacing2D = linear_filter.GetOutput().GetSpacing()
        
        spacing3D = [spacing2D[0], spacing2D[1], 1]
        vid_img = itk.GetImageFromArray(vid_linear.astype(np.float32))
        vid_img.SetSpacing(spacing3D)
        
        return vid_img
    
    def linearize_image(self, img, depth, zoom, offsetY, interpolate=True):
        filename = path.join(path.dirname(__file__), 'linearization_maps_sonosite', f'linear_map_depth{str(depth)}.npy')
        mapping = np.load(filename)
    
        frame_size = np.shape(mapping)[:2]
        img_linear = np.empty((frame_size[0],frame_size[1]))
    
        source_coords = mapping[:,:,:2].astype(int)
        source_coords_list = source_coords[:,:,::-1].flatten().tolist()
        kernels_list = mapping[:,:,2:].flatten().astype(float).tolist()
    
        self.unzoom_video(vid, zoom, offsetY)
    
        ImageType = itk.Image[itk.F,2]
        linear_filter = itkResampleImageUsingMapFilter[ImageType,ImageType].New()
        linear_filter.SetOutputSize([frame_size[1],frame_size[0]])
        linear_filter.SetSourceMapping(source_coords_list)
        linear_filter.SetKernels(kernels_list)
        linear_filter.SetInterpolate(interpolate)
        linear_filter.SetInput(img)
        linear_filter.Update()
        img_linear = linear_filter.GetOutput()
        return img_linear
