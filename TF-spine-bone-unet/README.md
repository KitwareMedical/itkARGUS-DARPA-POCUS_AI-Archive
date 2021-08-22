# Spine-Bone-Unet

This folder contains work related to reproducing the results of [Automatic Spine Ultrasound Segmentation for Scoliosis Visualization and Measurement](https://ieeexplore.ieee.org/document/9034149). In particular, modified versions of the Jupyter notebooks found [here](https://github.com/SlicerIGT/aigt/blob/master/Notebooks/Segmentation/SagittalSpineSegmentationStudy.ipynb) and [here](https://github.com/SlicerIGT/aigt/blob/master/Notebooks/Segmentation/SagittalSpineSegmentationTest.ipynb). 

TF1 contains notebooks that depend on TensorFlow 1, while TF2 contains notebooks that have been made to work with TensorFlow 2. Keras is also required as a separate dependency from TensorFlow. @tapphughesn has a Docker image that can be used the run the TF1 notebooks. 
TODO:
- Add dockerfile for TF1 docker image
