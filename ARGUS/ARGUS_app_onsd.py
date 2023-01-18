import os

from ARGUS_onsd_ar_inference import ARGUS_onsd_ar_inference
from ARGUS_onsd_roi_inference import ARGUS_onsd_roi_inference

class ARGUS_app_onsd:
    def __init__(self, device_num=0):
        self.onsd_ar = ARGUS_onsd_ar_inference(
            config_file_name="ARGUS_onsd_ar.cfg",
            network_name="final",
            device_num=device_num
        )
        self.onsd_roi = ARGUS_onsd_roi_inference()
        
        onsd_ar_best_models = [0, 0, 0]
        for r in range(self.onsd_ar.num_models):
            model_name = os.path.join(
                "Models",
                "onsd_ar_run"+str(r),
                "best_model_"+str(onsd_ar_best_models[r])+".pth"
            )
            self.onsd_ar.load_model(r, model_name)

        self.labels = None

        self.result = 0
        self.confidence = [0, 0]
            
    def ar_preprocess(self, vid_img):
        self.onsd_ar.volume_preprocess(vid_img)
        
    def ar_inference(self):
        self.labels = self.onsd_ar.volume_inference(step=5)
        
    def roi_inference(self):
        # First compute where the nerve is during the video (approximately)
        tmp_result, tmp_confidence, self.slice_min, self.slice_max = self.onsd_roi.inference(
                self.onsd_ar.input_image,
                self.labels)
        
        # Then compute the nerve in the relevent frames
        for i in range(len(self.slice_min)):
            self.labels = self.onsd_ar.volume_inference(
                step=1,
                slice_min=self.slice_min[i],
                slice_max=self.slice_max[i],
                use_cache=True
            )
        
        # Then estimates its width
        self.result,self.confidence, self.slice_min, self.slice_max = self.onsd_roi.inference(
                self.onsd_ar.input_image,
                self.labels)
        
    def decision(self):
        return self.result, self.confidence
