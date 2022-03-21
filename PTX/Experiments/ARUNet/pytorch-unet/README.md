## pytorch-unet

This directory will hold the pytorch version of the unet we will use in our pipeline, as well as all python scripts concerned with the unet. PyTorch/Monai will be used.

### Contents

A notebook with a filename like \*Study\*.ipynb contains code relating to loading data, training the UNet, monitoring with TensorBoard, and doing some evaluation.

A notebook with a filename like \*Test\*.ipynb contains code relating to testing and evaluting a model.

dataset_loader.py is the PyTorch DataSet object that is responsible for batching and augmenting data.

A notebook with a filename like get_timings\*.ipynb contains code relating to UNet inference timing.

pytorch_unet.py contains the main code for the UNet classes we are using

test_unet\*.py is unstructured code for testing the UNet

testing_stuff.ipynb is an unstructured notebook I used to investigate some things

unet_components.py contains some important modules (torch.nn.module) that the UNets use

### TODO

- write scripts for testing different training strategies
  - special kernel weights loss (domain knowledge) since data is correlated in axis 0 but not so much axis 1
  - downsample images to rectangles to aggregate information in axis 0 closer together
- write training, testing, and timing code for the temporal UNet.
  - Subtask: create a training dataset for the temporal UNet (from Slicer mrbs?)
