## pytorch-unet

This directory will hold the pytorch version of the unet we will use in our pipeline, as well as all python scripts concerned with the unet. PyTorch and monai will be used.

### TODO

- fully reproduce Spine Segmentation Unet (Tamas Ungi) results with the PyTorch version of the network
- write scripts for testing different training strategies
  - rectangular kernels
  - special kernel weights loss (domain knowledge) since data is correlated in axis 0 but not so much axis 1
  - downsample images to rectangles to aggregate information in axis 0 closer together
