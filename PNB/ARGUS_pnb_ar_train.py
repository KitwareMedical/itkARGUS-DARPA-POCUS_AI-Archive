import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_train import ARGUS_segmentation_train
from ARGUS_preprocess_butterfly import ARGUS_preprocess_butterfly

class ARGUS_pnb_ar_train(ARGUS_segmentation_train):
    
    def __init__(self, config_file_name="ARGUS_pnb_ar.cfg", network_name="final", device_num=0):
        
        super().__init__(config_file_name, network_name, device_num)
        
        self.preprocessed_pnb_video = []
        self.preprocess_pnb = ARGUS_preprocess_butterfly()
    
    def preprocess(self, vid):
        self.preprocessed_pnb_video = self.preprocess_pnb.process(vid)
        super(ARGUS_train_pnb_ar, self).preprocess_pnb(self.preprocessed_pnb_video)
        
    def preprocess_training(self, vid):
        self.preprocess_pnb_video = self.preprocess_pnb.process(vid)
        
        ar_input_array = np.empty([1, 1, self.num_channels, vid.shape[0], vid.shape[1]])

        input_array = self.SaptialResize(self.preprocessed_pnb_video)
        
        input_array_scaled = self.IntensityScale(input_array)
        
        return input_array_scaled
