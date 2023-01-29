import os

from ARGUS_pnb_ar_inference import ARGUS_pnb_ar_inference
from ARGUS_pnb_roi_inference import ARGUS_pnb_roi_inference

class ARGUS_app_pnb:
    def __init__(self, argus_dir=".", device_num=None, source=None):
        self.pnb_ar = ARGUS_pnb_ar_inference(
            config_file_name=os.path.join(argus_dir,"ARGUS_pnb_ar.cfg"),
            network_name="final",
            device_num=device_num,
            source=source
        )
        self.pnb_roi = ARGUS_pnb_roi_inference()
        
        pnb_ar_best_models = [8, 0, 7]
        for r in range(self.pnb_ar.num_models):
            model_name = os.path.join(
                argus_dir,
                "Models",
                "pnb_vfold_run"+str(r),
                "best_model_"+str(pnb_ar_best_models[r])+".pth"
            )
            self.pnb_ar.load_model(r, model_name)

        self.labels = None

        self.result = 0
        self.confidence = [0, 0]
            
    def ar_preprocess(self, vid_img):
        self.pnb_ar.preprocess(vid_img)
        
    def ar_inference(self):
        self.labels = self.pnb_ar.inference()
        
    def roi_inference(self):
        self.result,self.confidence = self.pnb_roi.inference(
                self.pnb_ar.input_image,
                self.labels)
        
    def decision(self):
        return self.result, self.confidence
