# ARGUS

* itkARGUS contains ITK methods written in C++ for accelerated performance.  Instructions for compiling itkARGUS are in that subfolder.  Those methos are automatically wrapped for python.
    * ResampleImageUsingMapFilter: contains code for resampling one image into another using a pre-defined mapping. Used to resample convex POCUS images into a linear space: transducer line -vs - depth.

* ARGUS-LinearAR-PTX is a workflow involving anatomic reconstruction using linearized b-mode images.

# Dependencies

* FFMPEG

On windows,
> conda install ffmpeg
> pip install ffmpeg-python
