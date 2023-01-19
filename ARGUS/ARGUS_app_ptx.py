import os

from ARGUS_ptx_ar_inference import ARGUS_ptx_ar_inference
from ARGUS_ptx_roi_inference import ARGUS_ptx_roi_inference

class ARGUS_app_ptx:
    def __init__(self, argus_dir=".", device_num=0):
        self.ptx_ar = ARGUS_ptx_ar_inference(
            config_file_name=os.path.join(argus_dir, "ARGUS_ptx_ar.cfg"),
            network_name="final",
            device_num=device_num
        )
        self.ptx_roi = ARGUS_ptx_roi_inference(
            config_file_name=os.path.join(argus_dir, "ARGUS_ptx_roi.cfg"),
            network_name="final",
            device_num=device_num
        )
        
        ptx_ar_best_models = [9, 8, 6]
        ptx_roi_best_models = [9, 8, 9]
        for r in range(self.ptx_ar.num_models):
            model_name = os.path.join(
                argus_dir,
                "Models",
                "ptx_ar_run"+str(r),
                "best_model_"+str(ptx_ar_best_models[r])+".pth"
            )
            self.ptx_ar.load_model(r, model_name)

        for r in range(self.ptx_roi.num_models):
            model_name = os.path.join(
                argus_dir,
                "Models",
                "ptx_roi_run"+str(r),
                "best_model_"+str(ptx_roi_best_models[r])+".pth"
            )
            self.ptx_roi.load_model(r, model_name)
            
        self.result = 0
        self.confidence = [0, 0]
            
    def ar_preprocess(self, vid_img):
        self.ptx_ar.preprocess(vid_img)
        
    def ar_inference(self):
        labels = self.ptx_ar.inference()
        
    def roi_generate_roi(self):
        self.ptx_roi.generate_roi(
            self.ptx_ar.input_image,
            self.ptx_ar.input_array,
            self.ptx_ar.label_array)
        
    def roi_inference(self):
        self.result,self.confidence = self.ptx_roi.inference()
        
    def decision(self):
        return self.result, self.confidence
