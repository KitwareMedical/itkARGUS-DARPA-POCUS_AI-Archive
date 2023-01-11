from os import path

import itk

from ARGUS_Timing import *
from ARGUS_IO import *
from ARGUS_app_taskid import ARGUS_app_taskid
from ARGUS_app_ptx import ARGUS_app_ptx
#from ARGUS_pnb import ARGUS_pnb
#from ARGUS_onsd import ARGUS_onsd
#from ARGUS_ett import ARGUS_ett

class ARGUS_app_ai:
    def __init__(self):
        device = torch.device('cpu')

        self.taskid = ARGUS_app_taskid(device_num)
        self.ptx = ARGUS_app_ptx(device_num)
        #self.pnb = ARGUS_pnb(model_dir)
        #self.onsd = ARGUS_onsd(model_dir)
        #self.ett = ARGUS_ett(model_dir)

    def predict(self, filename, debug=False, stats=None):
        time_this = ARGUS_time_this
        if stats:
            time_this = stats.time
    
        taskid = 0
        taskid_confidence = [0,0,0,0]
        taskname = [ "PTX", "PNB", "ONSD", "ETT" ]
        
        decision = 0
        decision_confidence = [0, 0]
        
        with time_this("all"):
            with time_this("Read Video"):
                us_video_img = ARGUS_load_video(filename)

                with time_this("Task Id"):
                    self.taskid.preprocess(us_video_img)
                    self.taskid.inference()
                    taskid,taskid_confidence = self.taskid.decision(us_video_img)

                with time_this("AR Preprocess"):
                    if taskid == 0:  #PTX
                        self.ptx.ar_preprocess(us_video)
                    elif taskid == 1: # PNB
                        # self.pnb.ar_preprocess(us_video)
                    elif taskid == 2: # ONSD
                        # self.onsd.ar_preprocess(us_video)
                    elif taskid == 3: # ETT
                        # self.ett.ar_preprocess(us_video)

            with time_this("Process Video"):
                with time_this("AR Inference Time:"):
                    if taskid == 0:  #PTX
                        self.ptx.ar_inference()
                    elif taskid == 1: # PNB
                        # self.pnb.ar_inference()
                    elif taskid == 2: # ONSD
                        # self.onsd.ar_inference()
                    elif taskid == 3: # ETT
                        # self.ett.ar_inference()

                with time_this("ROI Preprocess:"):
                    if taskid == 0:  #PTX
                        self.ptx.roi_generate_roi()
                    elif taskid == 1: # PNB
                        # self.pnb.roi_generate_roi()
                    elif taskid == 2: # ONSD
                        # self.onsd.roi_generate_roi()
                    elif taskid == 3: # ETT
                        # Nothing to do

                with time_this("ROI Inference Time:"):
                    if taskid == 0:  #PTX
                        self.ptx.roi_inference()
                    elif taskid == 1: # PNB
                        # self.pnb.roi_inference()
                    elif taskid == 2: # ONSD
                        # self.onsd.roi_inference()
                    elif taskid == 3: # ETT
                        # Nothing to do

                with time_this("Decision Time:"):
                    if taskid == 0:  #PTX
                        decision,decision_confidence = self.ptx.decision()
                    elif taskid == 1: # PNB
                        # decision,decision_confidence = self.pnb.decision()
                    elif taskid == 2: # ONSD
                        # decision,decision_confidence = self.onsd.decision()
                    elif taskid == 3: # ETT
                        # decision,decision_confidence = self.ett.decision()
        return dict(
            decision = decision,
            # debug info
            task_name = taskname[taskid],
            task_confidencePTX = taskid_confidence[0],
            task_confidencePNB = taskid_confidence[1],
            task_confidenceONSD = taskid_confidence[2],
            task_confidenceETT = taskid_confidence[3],
            pos_confidence = decision_confidence[0],
            neg_confidence = decision_confidence[1],
        )
