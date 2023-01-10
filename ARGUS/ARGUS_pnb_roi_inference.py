import site
site.addsitedir("../ARGUS")

from ARGUS_classification_inference import ARGUS_classification_inference

class ARGUS_pnb_roi(ARGUS_classification_inference):
    
    def __init__(self, config_file_name="ARGUS_pnb_roi.cfg", network_name="final", device_num=0):
        super().__init__(config_file_name, network_name, device_num)
        
    def generate_roi(self, vid):
        self.preprocessed_pnb_video = self.preprocess_pnb.process(vid)
        super().preprocess(self.preprocessed_pnb_video)
