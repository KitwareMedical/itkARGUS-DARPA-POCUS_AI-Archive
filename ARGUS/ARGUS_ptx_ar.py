import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_inference import ARGUS_segmentation_inference
from ARGUS_preprocess_ptx import ARGUS_preprocess_ptx

class ARGUS_ptx_ar(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_ptx_ar.cfg", network_name="final", device_num=0):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_ptx_video = []
        self.preprocess_ptx = ARGUS_preprocess_ptx()
    
    def preprocess(self, vid):
        self.preprocessed_ptx_video = self.preprocess_ptx.process(vid)
        super().preprocess(self.preprocessed_ptx_video)
