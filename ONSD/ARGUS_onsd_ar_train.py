import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_train import ARGUS_segmentation_train
from ARGUS_preprocess_butterfly import ARGUS_preprocess_butterfly

class ARGUS_onsd_ar_train(ARGUS_segmentation_train):
    
    def __init__(self, config_file_name="../ARGUS/ARGUS_onsd_ar.cfg", network_name="final", device_num=0):
        
        super().__init__(config_file_name, network_name, device_num)
        
        self.preprocessed_onsd_video = []
        self.preprocess_onsd = ARGUS_preprocess_butterfly()
    
    def preprocess(self, vid):
        self.preprocessed_onsd_video = self.preprocess_onsd.process(vid)
        super().preprocess(self.preprocessed_onsd_video)
        
    def preprocess_training(self, vid):
        self.preprocess_onsd_video = self.preprocess_onsd.process(vid)
        
        ar_input_array = np.empty([1, 1, self.num_channels, self.size_x, self.size_y])

        input_array = self.SaptialResize(self.preprocessed_onsd_video)
        
        input_array_scaled = self.IntensityScale(input_array)

        ar_input_array[0,0] = input_array_scaled
        
        return ar_input_array
