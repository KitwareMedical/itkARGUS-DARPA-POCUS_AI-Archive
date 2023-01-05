import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_inference import ARGUS_segmentation_inference
from ARGUS_preprocess_butterfly import ARGUS_preprocess_butterfly

class ARGUS_pnb_ar(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_pnb_ar.cfg", network_name="final", device_num=0):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_pnb_video = []
        self.preprocess_pnb = ARGUS_preprocess_butterfly()
        
    def preprocess(self, vid):
        self.preprocessed_pnb_video = self.preprocess_pnb.process(vid)
        super().preprocess(self.preprocessed_pnb_video)
