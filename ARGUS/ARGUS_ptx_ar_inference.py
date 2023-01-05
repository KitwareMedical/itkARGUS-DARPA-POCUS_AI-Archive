import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_inference import ARGUS_segmentation_inference
from ARGUS_preprocess_sonosite import ARGUS_preprocess_sonosite

class ARGUS_ptx_ar_inference(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_ptx_ar.cfg", network_name="final", device_num=0):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_sonosite_video = []
        self.preprocess_sonosite = ARGUS_preprocess_sonosite()
    
    def preprocess(self, vid, slice_num=None, scale_data=True, rotate_data=True):
        self.preprocessed_sonosite_video = self.preprocess_sonosite.process(vid)
        super().preprocess(self.preprocessed_sonosite_video, slice_num, scale_data, rotate_data)
