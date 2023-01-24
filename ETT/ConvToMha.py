import os

import argparse

import numpy as np

import itk
from itk import TubeTK as tube

import ffmpeg
import av

def ARGUS_shape_video(filename):
    p = ffmpeg.probe(filename, select_streams='v');
    width = p['streams'][0]['width']
    height = p['streams'][0]['height']
    return height, width

def ARGUS_load_video(filename):
    container = None
    try:
        container = av.open(filename)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        num_frames = stream.frames
        
        framerate = stream.average_rate
        
        frames = None
        for i,frame in enumerate(container.decode(stream)):
            if i == 0:
                frames = np.empty((num_frames, frame.height, frame.width))
            frames[i] = frame.to_ndarray(format='gray')
            
        vid = itk.GetImageFromArray(frames.astype(np.float32))
        spacing = [1, 1, 1.0/framerate]
        vid.SetSpacing(spacing)
        
        return vid
    finally:
        if container:
            container.close()

def prepare_argparser():
    parser = argparse.ArgumentParser(description='Video Converter')
    parser.add_argument('filename',
                        help='Video file to convert.')
    return parser

if __name__ == '__main__':
    parser = prepare_argparser()
    args = parser.parse_args()

    img = ARGUS_load_video(args.filename)

    immath = tube.ImageMath.New(img)
    arr = itk.GetArrayViewFromImage(img)
    if arr.max() > 1:
        immath.IntensityWindow(0,255,0,1)

    new_filename = os.path.basename(args.filename)[:-3]+".mha"

    itk.imwrite(immath.GetOutput(), new_filename, compression=True)
