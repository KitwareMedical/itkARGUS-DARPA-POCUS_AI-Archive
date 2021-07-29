from mmdet.apis import init_detector, inference_detector
import mmcv
from timerit import Timerit

config_file = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# download the checkpoint from model zoo and put it in checkpoints/
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

device = 'cpu'
img_fn = 'data/pocus_images/convex/Cov_Oliviera_2020_Fig4A.jpg'
video_fn = 'data/pocus_videos/linear/Cov-Atlas-pleural.mp4'


# faster rcnn r50 fpn

checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file, checkpoint_file, device=device)

# %timeit inference_detector(model, img_fn)
# 98.2 ms ± 2.17 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

print("running code: " + "inference_detector(model, img_fn)")
for _ in Timerit(num=10, verbose=2):
    inference_detector(model, img_fn)

# Test video
# %timeit -n10 inference_detector(model, mmcv.VideoReader(video_fn)[0])
# 113 ms ± 1.93 ms per loop (mean ± std. dev. of 7 runs, 20 loops each)

print("running code: " + "inference_detector(model, mmcv.VideoReader(video_fn)[0])")
for _ in Timerit(num=10,verbose=2):
    inference_detector(model, mmcv.VideoReader(video_fn)[0])

