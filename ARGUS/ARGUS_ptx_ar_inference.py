from ARGUS_segmentation_inference import ARGUS_segmentation_inference

from ARGUS_preprocess_butterfly import ARGUS_preprocess_butterfly
from ARGUS_preprocess_sonosite import ARGUS_preprocess_sonosite
from ARGUS_preprocess_clarius import ARGUS_preprocess_clarius

class ARGUS_ptx_ar_inference(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_ptx_ar.cfg", network_name="final", device_num=0, source=None):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_ptx_video = []
        if source=="Butterfly":
            self.preprocess_ptx = ARGUS_preprocess_butterfly(new_size=[self.size_x, self.size_y])
        elif source=="Sonosite" or source==None:
            self.preprocess_ptx = ARGUS_preprocess_sonosite(new_size=[self.size_x, self.size_y])
        elif source=="Clarius":
            print("Adjusting priors for Clarius")
            self.class_prior = [1, 0.9, 1]
            self.preprocess_ptx = ARGUS_preprocess_clarius(new_size=[self.size_x, self.size_y])
            
            
    def preprocess(self, vid_img, lbl=None, slice_num=None, crop_data=True, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_ptx_video = self.preprocess_ptx.process(vid_img)
        else:
            self.preprocessed_ptx_video = vid_img
        super().preprocess(self.preprocessed_ptx_video, lbl, slice_num, scale_data, rotate_data)
