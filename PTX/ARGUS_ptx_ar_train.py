import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_train import ARGUS_segmentation_train
from ARGUS_preprocess_ptx import ARGUS_preprocess_ptx

class ARGUS_ptx_ar_train(ARGUS_segmentation_train):
    
    def __init__(self, config_file_name="ARGUS_ptx_ar.cfg", network_name="final", device_num=0):
        
        super().__init__(config_file_name, network_name, device_num)
        
        self.preprocessed_ptx_video = []
        self.preprocess_ptx = ARGUS_preprocess_ptx()
    
    def preprocess(self, vid):
        self.preprocessed_ptx_video = self.preprocess_ptx.process(vid)
        super(ARGUS_train_ptx_ar, self).preprocess_ptx(self.preprocessed_ptx_video)
        
    def preprocess_training(self, vid):
        self.preprocess_ptx_video = self.preprocess_ptx.process(vid)
        
        ar_input_array = np.empty([1, 1, self.num_channels, vid.shape[0], vid.shape[1]])

        input_array = self.SaptialResize(self.preprocessed_ptx_video)
        
        input_array_scaled = self.IntensityScale(input_array)
        
        return input_array_scaled
