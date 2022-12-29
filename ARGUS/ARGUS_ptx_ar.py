import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_inference import ARGUS_segmentation_inference
from ARGUS_linearization_ptx import ARGUS_linearization_ptx

class ARGUS_ptx_ar(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_ptx_ar.cfg", network_name="final", device_num=0):
        
        super().__init__(config_file_name, network_name, device_num)
        
        self.linearized_video = []
        self.linearize = ARGUS_linearization_ptx()
    
    def preprocess(self, vid):
        self.linearized_video = self.linearize.process(vid)
        super(ARGUS_train_ptx_ar, self).preprocess(self.linearized_video)
