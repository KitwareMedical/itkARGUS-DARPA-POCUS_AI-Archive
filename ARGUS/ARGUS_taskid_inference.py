import site
site.addsitedir("../ARGUS")

from ARGUS_classification_inference import ARGUS_classification_inference
from ARGUS_linearization_ptx import ARGUS_linearization_ptx

class ARGUS_taskid(ARGUS_classification_inference):
    def __init__(self, config_file_name="ARGUS_taskid.cfg", network_name="final", device_num=0):
        super().__init__(self, config_file, network_name, device_num)