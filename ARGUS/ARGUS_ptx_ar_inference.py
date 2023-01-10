import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_inference import ARGUS_segmentation_inference
from ARGUS_preprocess_sonosite import ARGUS_preprocess_sonosite

class ARGUS_ptx_ar_inference(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_ptx_ar.cfg", network_name="final", device_num=0):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_sonosite_video = []
        self.preprocess_sonosite = ARGUS_preprocess_sonosite()
    
    def preprocess(self, vid, lbl=None, slice_num=None, crop_data=True, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_sonosite_video = self.preprocess_sonosite.process(vid)
        else:
            self.preprocessed_sonosite_video = vid
        super().preprocess(self.preprocessed_sonosite_video, lbl, slice_num, crop_data, scale_data, rotate_data)
