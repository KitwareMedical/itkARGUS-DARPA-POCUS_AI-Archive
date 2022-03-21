## Object detection models timing

This folder contains work I did to time mmdetection models on the SurfaceBook 2 with Windows (using WSL2)

See google doc here for steps I took to download mmdet: https://docs.google.com/document/d/1ZFxwfB6ZYVWBj4Q2GrgLIJ97ej-IgK5ErBF4VBU9uFI/edit?usp=sharing

#### Setup

- `git clone https://github.com/jannisborn/covid19_ultrasound.git`
- `git clone https://github.com/open-mmlab/mmdetection.git`
- `cp -r covid19_ultrasound/data .`
- `mkdir checkpoints`
- Download these files and put them in `checkpoints`
	- https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth 
	-  https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
