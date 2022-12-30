from os import path

import itk

from ARGUS_Timing import *
from ARGUS_IO import *
from ARGUS_taskid import ARGUS_taskid
#from ARGUS_ptx import ARGUS_ptx
#from ARGUS_pnb import ARGUS_pnb
#from ARGUS_onsd import ARGUS_onsd
#from ARGUS_ett import ARGUS_ett

class ARGUS_app_ai:
    def __init__(self, device_name='cpu', model_dir='Models'):
        device = torch.device(device_name)

        self.taskid = ARGUS_taskid(device, model_dir)
        #self.ptx = ARGUS_ptx(model_dir)
        #self.pnb = ARGUS_pnb(model_dir)
        #self.onsd = ARGUS_onsd(model_dir)
        #self.ett = ARGUS_ett(model_dir)

    def predict(self, filename, debug=False, stats=None):
        time_this = ARGUS_time_this
        if stats:
            time_this = stats.time
    
        with time_this("all"):
            with time_this("Read Video"):
                us_video = ARGUS_load_video(filename)

                with time_this("Task Id"):
                    taskid = self.taskid.get_taskid(us_video)

                with time_this("AR Preprocess"):
                    if taskid == 0:  #PTX
                        # self.ptx.ar_preprocess(us_video)
                    elif taskid == 1: # PNB
                        # self.pnb.ar_preprocess(us_video)
                    elif taskid == 2: # ONSD
                        # self.onsd.ar_preprocess(us_video)
                    elif taskid == 3: # ETT
                        # self.ett.ar_preprocess(us_video)

            with time_this("Process Video"):
                with time_this("AR Inference Time:"):
                    if taskid == 0:  #PTX
                        # self.ptx.ar_inference()
                    elif taskid == 1: # PNB
                        # self.pnb.ar_inference()
                    elif taskid == 2: # ONSD
                        # self.onsd.ar_inference()
                    elif taskid == 3: # ETT
                        # self.ett.ar_inference()

                with time_this("ROI Preprocess:"):
                    if taskid == 0:  #PTX
                        # self.ptx.roi_preprocess()
                    elif taskid == 1: # PNB
                        # self.pnb.roi_preprocess()
                    elif taskid == 2: # ONSD
                        # self.onsd.roi_preprocess()
                    elif taskid == 3: # ETT
                        # Nothing to do

                with time_this("ROI Inference Time:"):
                    if taskid == 0:  #PTX
                        # self.ptx.roi_inference()
                    elif taskid == 1: # PNB
                        # self.pnb.roi_inference()
                    elif taskid == 2: # ONSD
                        # self.onsd.roi_inference()
                    elif taskid == 3: # ETT
                        # Nothing to do

                with time_this("Decision Time:"):
                    if taskid == 0:  #PTX
                        # decision,pos_confidence,neg_confidence = self.ptx.decision()
                    elif taskid == 1: # PNB
                        # decision,pos_confidence,neg_confidence = self.pnb.decision()
                    elif taskid == 2: # ONSD
                        # decision,pos_confidence,neg_confidence = self.onsd.decision()
                    elif taskid == 3: # ETT
                        # decision,pos_confidence,neg_confidence = self.ett.decision()


        if taskid == 0:  #PTX
            taskname = "PTX"
        elif taskid == 1: # PNB
            taskname = "PNB"
        elif taskid == 2: # ONSD
            taskname = "ONSD"
        elif taskid == 3: # ETT
            taskname = "ETT"

        return dict(
            decision=decision,
            # debug info
            task_name=taskname,
            pos_confidence=pos_confidence,
            neg_confidence=neg_confidence,
        )
