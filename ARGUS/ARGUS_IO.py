import itk

import numpy as np

import ffmpeg
import av

def ARGUS_shape_video(filename):
    p = ffmpeg.probe(filename, select_streams='v');
    width = p['streams'][0]['width']
    height = p['streams'][0]['height']
    return height, width

def ARGUS_load_video(filename, frame_limit=None):
    vid = None
    container = None
    try:
        container = av.open(filename)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        min_frame = 0
        num_frames = stream.frames
        if frame_limit != None and num_frames > frame_limit:
            min_frame = num_frames-frame_limit
            num_frames = frame_limit
        
        framerate = stream.average_rate
        
        frames = None
        for i,frame in enumerate(container.decode(stream)):
            if i == min_frame:
                frames = np.empty((num_frames, frame.height, frame.width),dtype=np.float32)
            if i >= min_frame:
                frames[i-min_frame] = frame.to_ndarray(format='gray').astype(np.float32)
            
        vid = itk.GetImageViewFromArray(frames)
        spacing = [1, 1, 1.0/framerate]
        vid.SetSpacing(spacing)
    finally:
        if container:
            container.close()
            
    return vid