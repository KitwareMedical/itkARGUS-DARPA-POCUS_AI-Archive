from ARGUS_Timing import *
from ARGUS_IO import *
from ARGUS_app_taskid import ARGUS_app_taskid
from ARGUS_app_ptx import ARGUS_app_ptx
from ARGUS_app_pnb import ARGUS_app_pnb
from ARGUS_app_onsd import ARGUS_app_onsd
#from ARGUS_app_ett import ARGUS_app_ett

class ARGUS_app_ai:
    def __init__(self):
        device_num = 0
        
        self.taskid = ARGUS_app_taskid(device_num)
        self.ptx = ARGUS_app_ptx(device_num)
        self.pnb = ARGUS_app_pnb(device_num)
        self.onsd = ARGUS_app_onsd(device_num)
        #self.ett = ARGUS_app_ett(device_num)

    def predict(self, filename, debug=False, stats=None, taskid=None):
        time_this = ARGUS_time_this
        if stats:
            time_this = stats.time
    
        taskid_confidence = [0,0,0,0]
        taskname = [ "PTX", "PNB", "ONSD", "ETT" ]
        
        decision = 0
        decision_confidence = [0, 0]
        
        with time_this("all"):
            with time_this("Read Video"):
                with time_this("Read Video: Read from disk"):
                    us_video_img = ARGUS_load_video(filename)
                    video_time = (
                        us_video_img.GetLargestPossibleRegion().GetSize()[2] *
                        us_video_img.GetSpacing()[2]
                    )
                    print(f"Length of video = {video_time} seconds")

                if taskid == None:
                    with time_this("Read Video: Task Id"):
                        self.taskid.preprocess(us_video_img)
                        self.taskid.inference()
                        taskid,taskid_confidence = self.taskid.decision()

            with time_this("Preprocess Video"):
                with time_this("Preprocess for AR"):
                    if taskid == 0:  #PTX
                        print("PTX")
                        self.ptx.ar_preprocess(us_video_img)
                    elif taskid == 1: # PNB
                        print("PNB")
                        self.pnb.ar_preprocess(us_video_img)
                    elif taskid == 2: # ONSD
                        print("ONSD")
                        self.onsd.ar_preprocess(us_video_img)
                    elif taskid == 3: # ETT
                        print("ETT")
                        # self.ett.ar_preprocess(us_video_img)

                with time_this("Preprocess AR Inference"):
                    if taskid == 0:  #PTX
                        self.ptx.ar_inference()
                    elif taskid == 1: # PNB
                        self.pnb.ar_inference()
                    elif taskid == 2: # ONSD
                        self.onsd.ar_inference()
                    #elif taskid == 3: # ETT
                        # self.ett.ar_inference()

                with time_this("Preprocess for ROI"):
                    if taskid == 0:  #PTX
                        self.ptx.roi_generate_roi()
                    #elif taskid == 1: # PNB
                        # Nothing to do
                    #elif taskid == 2: # ONSD
                        # Nothing to do
                    #elif taskid == 3: # ETT
                        # Nothing to do

            with time_this("Process Video"):
                with time_this("Process Video: ROI Inference"):
                    if taskid == 0:  #PTX
                        self.ptx.roi_inference()
                    elif taskid == 1: # PNB
                        self.pnb.roi_inference()
                    elif taskid == 2: # ONSD
                        self.onsd.roi_inference()
                    #elif taskid == 3: # ETT
                        # Nothing to do

                with time_this("Process Video: Decision"):
                    if taskid == 0:  #PTX
                        decision,decision_confidence = self.ptx.decision()
                    elif taskid == 1: # PNB
                        decision,decision_confidence = self.pnb.decision()
                    elif taskid == 2: # ONSD
                        decision,decision_confidence = self.onsd.decision()
                    #elif taskid == 3: # ETT
                        # decision,decision_confidence = self.ett.decision()
        return dict(
            decision = decision,
            task_name = taskname[taskid],
            task_confidence_PTX = taskid_confidence[0],
            task_confidence_PNB = taskid_confidence[1],
            task_confidence_ONSD = taskid_confidence[2],
            task_confidence_ETT = taskid_confidence[3],
            decision_confidence_0 = decision_confidence[0],
            decision_confidence_1 = decision_confidence[1],
        )
