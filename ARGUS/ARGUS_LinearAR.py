import itk

from ARGUSUtils_Timing import *
from ARGUSUtils_IO import *
from ARGUSUtils_Linearization import *
from ARGUSUtils_ARNet import *
from ARGUSUtils_ROINet import *

def ARGUS_LinearAR(filename,device_name="cpu",debug=False):
    arnet_model_filename = "./Models/BAMC_PTX_ARUNET-3D-PR-Final15/best_model.vfold_0.pth"
    roinet_model_filename = "./Models/BAMC_PTX_ROINet-StdDevExtended-ExtrudedNS-Final15/best_model.vfold_0.pth"
    
    with time_this("Load Video:",debug):
        us_video = load_video(filename,0,0)

    with time_this("Linearization Video:",debug):
        us_video_linear = linearize_video(us_video).transpose([2,1,0])

    device = torch.device(device_name)

    with time_this("Load ARNet model",debug):
        arnet_model = arnet_load_model(arnet_model_filename,device)

    with time_this("Preprocess for ARNet",debug):
        arnet_input_tensor = arnet_preprocess_video(us_video_linear)

    if(debug):
        itk.imwrite(itk.GetImageFromArray(arnet_input_tensor[0,0,:,:,:]),
            "ARUNet_preprocessed_input.mha")

    with time_this("ARNet Inference Time:",debug):
        arnet_output = arnet_inference(arnet_input_tensor,arnet_model,
            device)
    
    if(debug):
        itk.imwrite(itk.GetImageFromArray(arnet_output),
            "ARUNet_output.mha")

    with time_this("ROI Extraction Time:",debug):
        roinet_input_roi = roinet_segment_roi(us_video_linear,arnet_output)

    if(debug):
        itk.imwrite(itk.GetImageFromArray(roinet_input_roi.astype(np.float32)),
            "ROINet_input_roi.mha")

    with time_this("Load ROINet model",debug):
        roinet_model = roinet_load_model(roinet_model_filename,device)


    with time_this("Preprocess for ROINet",debug):
        roinet_input_tensor = roinet_preprocess_roi(roinet_input_roi)

    if(debug):
        itk.imwrite(itk.GetImageFromArray(roinet_input_tensor[0,:,:,:]),
            "ROINet_preprocessed_input.mha")

    with time_this("ROINet Inference Time:",debug):
        decision,not_sliding_count,sliding_count,class_array = roinet_inference(
            roinet_input_tensor,roinet_model,device,True)

    if(debug):
        itk.imwrite( itk.GetImageFromArray(class_array),"ARGUS_output.mha")
        print(decision,not_sliding_count,sliding_count)

    return decision,not_sliding_count,sliding_count
