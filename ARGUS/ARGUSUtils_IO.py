import numpy as np

import ffmpeg
import av

def shape_video(filename):
    p = ffmpeg.probe(filename, select_streams='v');
    width = p['streams'][0]['width']
    height = p['streams'][0]['height']
    return height, width

def load_video(filename,height=0,width=0):
    if(height==0 or width==0):
        height,width = shape_video(filename)

    container = av.open(filename)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    
    num_frames = stream.frames
    
    frames = np.empty((num_frames,height,width))
    for i,frame in enumerate(container.decode(stream)):
        frames[i] = frame.to_ndarray(format='gray')
    return frames
