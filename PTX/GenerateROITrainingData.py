import os
import glob

import numpy as np

import monai
from monai.data import ITKWriter
from monai.transforms import LoadImage

import itk
from itk import TubeTK as tube

import site
site.addsitedir("../ARGUS")

from ARGUS_IO import *
from ARGUS_ptx_ar_train import ARGUS_ptx_ar_train
from ARGUS_ptx_roi_inference import ARGUS_ptx_roi_inference

device_num = 0
best_models = [9, 8, 6]

image_files = sorted(glob.glob("./Data_PTX/images/*.mha"))
label_files = sorted(glob.glob("./Data_PTX/labels/*.mha"))

ar_nnet = ARGUS_ptx_ar_train(
        "../ARGUS/ARGUS_ptx_ar.cfg",
        "vfold",
        device_num=device_num
        )

for r in range(len(best_models)):
    model_name = os.path.join(
            "./Results_Best",
            "ptx_vfold_run"+str(r),
            "best_model_"+str(best_models[r])+".pth"
            )
    ar_nnet.load_model( r, model_name )

roi_nnet = ARGUS_ptx_roi_inference(
        os.path.join( "..", "ARGUS", "ARGUS_ptx_roi.cfg"),
        "vfold",
        device_num=device_num
        )

for file_num in [0]: #range(len(image_files)):
    img = itk.imread(image_files[file_num])
    lbl = itk.imread(label_files[file_num], itk.SS)
    for slice_num in range(16,img.shape[0]-16,16):
        ar_nnet.preprocess(
            img,
            lbl_img=lbl,
            slice_num=slice_num, 
            crop_data=False, 
            scale_data=False,
            rotate_data=True
        )
        ar_in_image = ar_nnet.input_image
        ar_in_array = ar_nnet.input_array
        ar_out_ideal_array = ar_nnet.label_array
        roi_nnet.generate_roi(
            ar_in_image,
            ar_in_array,
            ar_out_ideal_array
        )
        roi_ideal_image = roi_nnet.input_image
        roi_ideal_array = roi_nnet.input_array
        roi_ideal_label_array = roi_net.label_array
        
        img_base = os.path.splitext(os.path.basename(image_files[file_num]))[0]
        lbl_base = os.path.splitext(os.path.basename(label_files[file_num]))[0]
        img_dir_basename = "./Data_PTX/roi_images"
        lbl_dir_basename = "./Data_PTX/roi_labels"
        
        new_img_basename = img_base+"_"+str(slice_num)+"_img.mha"
        new_arr_basename = img_base+"_"+str(slice_num)+"_arr.mha"
        new_lbl_basename = lbl_base+"_"+str(slice_num)+".mha"
        
        itk.imwrite(roi_ideal_image,
                    os.path.join(img_dir_basename, new_img_basename))
        itk.imwrite(itk.GetImageFromArray(roi_ideal_array),
                    os.path.join(img_dir_basename, new_arr_basename))
        itk.imwrite(itk.GetImageFromArray(roi_ideal_label_array),
                    os.path.join(lbl_dir_basename, new_lbl_basename))

        ar_out_pred_array = ar_nnet.inference()
        print("ar_pred", ar_pred.shape)
        roi_pred, roi_pred_label = roi_nnet.generate_roi(ar_input[0,0], ar_pred)
        Writer.set_data_array(roi_pred)
        Writer.write("test_roi_pred.mha")
        Writer.set_data_array(roi_ideal_label)
        Writer.write("test_roi_pred_label.mha")