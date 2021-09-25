import numpy as np

import ffmpeg

def shape_ffmpeg( filename ):
    p = ffmpeg.probe(filename, select_streams='v');
    width = p['streams'][0]['width']
    height = p['streams'][0]['height']
    return height, width

def load_ffmpeg( filename, height, width ):
    p = ffmpeg.probe(filename, select_streams='v');
    width = p['streams'][0]['width']
    height = p['streams'][0]['height']

    in_bytes, _ = (
        ffmpeg
        .input(filename)
        .video # Video only (no audio).
        .output('pipe:', format='rawvideo', pix_fmt='gray')  # Set the output format to raw video in 8 bit grayscale
        .run(capture_stdout=True)
    )
    n_frames = len(in_bytes) // (height*width)  # Compute the number of frames.
    frames = np.frombuffer(in_bytes, np.uint8).reshape(n_frames, height, width) # Reshape buffer to array of n_frames frames (shape of each frame is (height, width)).
    return frames
