from os import path

import itk

import ARGUSUtils_Timing
from ARGUSUtils_IO import *
from ARGUSUtils_Linearization import *
from ARGUSUtils_ARNet import *
from ARGUSUtils_ROINet import *

class ARGUS_LinearAR:
    def __init__(self, device_name='cpu', model_dir='Models'):
        self.device = torch.device(device_name)

        self.arnet_model_filename = path.join(model_dir, "BAMC_PTX_ARUNET-3D-PR-Final15", "Epoch333_model.pth")

        self.roinet_model0_filename = path.join(model_dir, "BAMC_PTX_ROINet-StdDevExtended-ExtrudedNS-Final15", "best_model.pth")
        self.roinet_model1_filename = path.join(model_dir, "BAMC_PTX_ROINet-StdDevExtended-ExtrudedNS-Final15", "Epoch333_model.pth")
        self.roinet_model2_filename = path.join(model_dir, "BAMC_PTX_ROINet-StdDevExtended-ExtrudedNS-Final15", "best_model.pth")

        self.arnet_model = arnet_load_model(self.arnet_model_filename, self.device)
        self.roinet_model0 = roinet_load_model(self.roinet_model0_filename,self.device)
        self.roinet_model1 = roinet_load_model(self.roinet_model1_filename,self.device)
        self.roinet_model2 = roinet_load_model(self.roinet_model2_filename,self.device)
    
    def predict(self, filename, debug=False, stats=None):
        time_this = ARGUSUtils_Timing.time_this
        if stats:
            time_this = stats.time
    
        with time_this("all"):
            with time_this("Read Video"):
                us_video = load_video(filename)

            with time_this("Process Video"):
                with time_this("Linearization Video"):
                    us_video_linear = linearize_video(us_video).transpose([2,1,0])

                with time_this("Preprocess for ARNet"):
                    arnet_input_tensor = arnet_preprocess_video(us_video_linear)

                if(debug):
                    itk.imwrite(itk.GetImageFromArray(arnet_input_tensor[0,0,:,:,:]),
                        "results/ARUNet_preprocessed_input.mha")

                with time_this("ARNet Inference Time:"):
                    arnet_output = arnet_inference(arnet_input_tensor,self.arnet_model,
                        self.device)
                
                if(debug):
                    itk.imwrite(itk.GetImageFromArray(arnet_output),
                        "results/ARUNet_output.mha")

                with time_this("ROI Extraction Time:"):
                    roinet_input_roi = roinet_segment_roi(us_video_linear,arnet_output)

                if(debug):
                    itk.imwrite(itk.GetImageFromArray(roinet_input_roi.astype(np.float32)),
                        "results/ROINet_input_roi.mha")

                with time_this("Preprocess for ROINet"):
                    roinet_input_tensor0 = roinet_preprocess_roi(roinet_input_roi, 0)
                    roinet_input_tensor1 = roinet_preprocess_roi(roinet_input_roi, 1)
                    roinet_input_tensor2 = roinet_preprocess_roi(roinet_input_roi, 2)

                if(debug):
                    itk.imwrite(itk.GetImageFromArray(roinet_input_tensor[0,:,:,:]),
                        "results/ROINet_preprocessed_input.mha")

                with time_this("ROINet Inference Time:"):
                    decision0,not_sliding_count0,sliding_count0,class_array0 = roinet_inference(
                        roinet_input_tensor0,self.roinet_model0,self.device,True)
                    decision1,not_sliding_count1,sliding_count1,class_array1 = roinet_inference(
                        roinet_input_tensor1,self.roinet_model1,self.device,True)
                    decision2,not_sliding_count2,sliding_count2,class_array2 = roinet_inference(
                        roinet_input_tensor2,self.roinet_model2,self.device,True)
                    ns = 0
                    not_sliding_count = not_sliding_count0
                    sliding_count = sliding_count0
                    if(decision0 == "Not Sliding"):
                        ns += 1
                    not_sliding_count += not_sliding_count1
                    sliding_count += sliding_count1
                    if(decision1 == "Not Sliding"):
                        ns += 1
                    not_sliding_count += not_sliding_count2
                    sliding_count += sliding_count2
                    if(decision2 == "Not Sliding"):
                        ns += 1

                    if(ns >= 2):
                        decision = "Not Sliding"
                    else:
                        decision = "Sliding"

                if(debug):
                    itk.imwrite( itk.GetImageFromArray(class_array0),
                        "results/ARGUS_output0.mha")
                    print(decision0,not_sliding_count0,sliding_count0)
                    itk.imwrite( itk.GetImageFromArray(class_array1),
                        "results/ARGUS_output1.mha")
                    print(decision1,not_sliding_count1,sliding_count1)
                    itk.imwrite( itk.GetImageFromArray(class_array2),
                        "results/ARGUS_output2.mha")
                    print(decision2,not_sliding_count2,sliding_count2)

        return dict(
            decision=decision,
            # debug info
            not_sliding_count=not_sliding_count,
            sliding_count=sliding_count,
            arnet_input_tensor=arnet_input_tensor,
            arnet_output=arnet_output,
            roinet_input_roi=roinet_input_roi,
            roinet_input_tensor=roinet_input_tensor0,
            class_array=class_array0,
        )
