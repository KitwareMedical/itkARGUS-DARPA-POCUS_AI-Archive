import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_train import ARGUS_segmentation_train
from ARGUS_preprocess_sonosite import ARGUS_preprocess_sonosite

class ARGUS_ptx_ar_train(ARGUS_segmentation_train):
    
    def __init__(self, config_file_name="ARGUS_ptx_ar.cfg", network_name="final", device_num=0):
        
        super().__init__(config_file_name, network_name, device_num)
        
        self.preprocessed_sonosite_video = []
        self.preprocess_sonosite = ARGUS_preprocess_sonosite()
    
    def preprocess(self, vid, lbl_img=None, slice_num=None, crop_data=True, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_sonosite_video = self.preprocess_sonosite.process(vid)
        else:
            self.preprocessed_sonosite_video = vid
        super().preprocess(self.preprocessed_sonosite_video, lbl_img=lbl_img, slice_num=slice_num, scale_data=scale_data, rotate_data=rotate_data)
        
    def preprocess_training(self, vid):
        self.preprocess_sonosite_video = self.preprocess_sonosite.process(vid)
        
        # Should do this using ITK methods to preserve spacing, origin, etc
        input_array = self.SaptialResize(self.preprocessed_sonosite_video)
        
        input_array_scaled = self.IntensityScale(input_array)
        
        return input_array_scaled
