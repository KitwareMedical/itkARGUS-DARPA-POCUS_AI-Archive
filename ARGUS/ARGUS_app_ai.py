from traceback import print_exc

from ARGUS_Timing import *
from ARGUS_IO import *
from ARGUS_app_taskid import ARGUS_app_taskid
from ARGUS_app_ptx import ARGUS_app_ptx
from ARGUS_app_pnb import ARGUS_app_pnb
from ARGUS_app_onsd import ARGUS_app_onsd
from ARGUS_app_ett import ARGUS_app_ett

class ARGUS_app_ai:
    tasks = [ "PTX", "PNB", "ONSD", "ETT" ]
    sources = [ "Sonosite", "Butterfly", "Clarius" ]
        
    def __init__(self, argus_dir=".", device_num=None, source=None):
        self.argus_dir = argus_dir
        self.device_num = device_num
        self.source = source
        
        self.taskid = ARGUS_app_taskid(argus_dir, device_num, source)
        self.ptx = ARGUS_app_ptx(argus_dir, device_num, source)
        self.pnb = ARGUS_app_pnb(argus_dir, device_num, source)
        self.onsd = ARGUS_app_onsd(argus_dir, device_num, source)
        self.ett = ARGUS_app_ett(argus_dir, device_num, source)
        
    def predict(self, filename, debug=False, stats=None, task=None, source=None):
        time_this = ARGUS_time_this
        if stats:
            time_this = stats.time
    
        task_confidence = [0,0,0,0]
        
        decision = 0
        decision_confidence = [0, 0]
        
        if self.source == None:
            print(f"ERROR: Please specify ultrasound source: {self.sources}")
            return None
        
        print("File:", filename)
        with time_this("all"):
            with time_this("Read Video"):
                with time_this("Read Video: Read from disk"):
                    try:
                        us_video_img = ARGUS_load_video(filename)
                    except:
                        print(f"ERROR: Could not load video {filename}")
                        print_exc(limit=0)
                        return None
                    video_time = (
                        us_video_img.GetLargestPossibleRegion().GetSize()[2] *
                        us_video_img.GetSpacing()[2]
                    )
                if task == None:
                    with time_this("Read Video: Task Id"):
                        try:
                            self.taskid.preprocess(us_video_img)
                            self.taskid.inference()
                            taskid,task_confidence = self.taskid.decision()
                            if taskid != None:
                                task = self.tasks[taskid]
                            else:
                                print(f"ERROR: Could not identify task.")
                                return None
                        except:
                            print(f"ERROR: task identification failed.")
                            print_exc(limit=0)
                            return None
                else:
                    if task in self.tasks:
                        taskid = self.tasks.index(task)
                    else:
                        print(f"ERROR: task {task} not defined.")
                        return None

            with time_this("Preprocess Video"):
                with time_this("Preprocess for AR"):
                    try:
                        if task == "PTX":
                            print("   Task: PTX")
                            self.ptx.ar_preprocess(us_video_img)
                        elif task == "PNB": 
                            print("   Task: PNB")
                            self.pnb.ar_preprocess(us_video_img)
                        elif task == "ONSD":
                            print("   Task: ONSD")
                            self.onsd.ar_preprocess(us_video_img)
                        elif task == "ETT":
                            print("   Task: ETT")
                    except:
                        print(f"ERROR: Could not preprocess for anatomic reconstruction.")
                        print_exc(limit=0)
                        return None
    
                with time_this("Preprocess AR Inference"):
                    try:
                        if task == "PTX":
                            self.ptx.ar_inference()
                        elif task == "PNB":
                            self.pnb.ar_inference()
                        elif task == "ONSD":
                            self.onsd.ar_inference()
                        #elif task == "ETT":
                            # Nothing to do
                    except:
                        print(f"ERROR: Could not run anatomic reconstruction inference.")
                        print_exc(limit=0)
                        return None

                with time_this("Preprocess for ROI"):
                    try:
                        if task == "PTX":
                            self.ptx.roi_generate_roi()
                        #elif task == "PNB":
                            # Nothing to do
                        #elif task == "ONSD":
                            # Nothing to do
                        elif task == "ETT":
                            self.ett.roi_preprocess(us_video_img)
                    except:
                        print(f"ERROR: Could not preprocess for decision inference.")
                        print_exc(limit=0)
                        return None

            with time_this("Process Video"):
                with time_this("Process Video: ROI Inference"):
                    try:
                        if task == "PTX":
                            self.ptx.roi_inference()
                        elif task == "PNB":
                            self.pnb.roi_inference()
                        elif task == "ONSD":
                            self.onsd.roi_inference()
                        elif task == "ETT":
                            self.ett.roi_inference()
                    except:
                        print(f"ERROR: Could not run decision inference.")
                        print_exc(limit=0)
                        return None

                with time_this("Process Video: Decision"):
                    try:
                        if task == "PTX":
                            decision,decision_confidence = self.ptx.decision()
                        elif task == "PNB":
                            decision,decision_confidence = self.pnb.decision()
                        elif task == "ONSD":
                            decision,decision_confidence = self.onsd.decision()
                        elif task == "ETT":
                             decision,decision_confidence = self.ett.decision()
                    except:
                        print(f"ERROR: Could not deliver decision.")
                        print_exc(limit=0)
                        return None

        print(f"   Prediction: {decision}")
        print(f"      Confidence Measure 0: {decision_confidence[0]}")
        print(f"      Confidence Measure 1: {decision_confidence[1]}")

        return dict(
            decision = decision,
            task_name = task,
            task_confidence_PTX = task_confidence[0],
            task_confidence_PNB = task_confidence[1],
            task_confidence_ONSD = task_confidence[2],
            task_confidence_ETT = task_confidence[3],
            decision_confidence_0 = decision_confidence[0],
            decision_confidence_1 = decision_confidence[1],
            video_length = video_time,
        )
