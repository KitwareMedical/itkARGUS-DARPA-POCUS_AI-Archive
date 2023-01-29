import itk

from ARGUS_segmentation_inference import ARGUS_segmentation_inference
from ARGUS_preprocess_butterfly import ARGUS_preprocess_butterfly
from ARGUS_preprocess_sonosite import ARGUS_preprocess_sonosite
from ARGUS_preprocess_clarius import ARGUS_preprocess_clarius

class ARGUS_pnb_ar_inference(ARGUS_segmentation_inference):
    
    def __init__(self, config_file_name="ARGUS_pnb_ar.cfg", network_name="final", device_num=0, source=None):
        super().__init__(config_file_name, network_name, device_num)
        self.preprocessed_pnb_video = []
        if source=="Butterfly" or source==None:
            self.preprocess_pnb = ARGUS_preprocess_butterfly(new_size=[self.size_x, self.size_y])
        elif source=="Sonosite":
            self.preprocess_pnb = ARGUS_preprocess_sonosite(new_size=[self.size_x, self.size_y])
        elif source=="Clarius":
            self.preprocess_pnb = ARGUS_preprocess_clarius(new_size=[self.size_x, self.size_y])
        
    def preprocess(self, vid, lbl=None, slice_num=None, crop_data=True, scale_data=True, rotate_data=True):
        if crop_data:
            self.preprocessed_pnb_video = self.preprocess_pnb.process( vid )
        else:
            self.preprocessed_pnb_video = vid
        super().preprocess(self.preprocessed_pnb_video, lbl, slice_num, scale_data, rotate_data)
