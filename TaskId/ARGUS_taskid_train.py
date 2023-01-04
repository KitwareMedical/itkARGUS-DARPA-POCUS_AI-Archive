import site
site.addsitedir("../ARGUS")

from ARGUS_classification_train import ARGUS_classification_train

class ARGUS_taskid_train(ARGUS_classification_train):
    def __init__(self, config_file_name="ARGUS_taskid.cfg", network_name="final", device_num=0):
        super().__init__(config_file_name, network_name, device_num)
