from experiments.ONSD.ARUNet_Nerve_Network import ARUNet_Nerve_Network
from utils.ARGUS_vfold_setup import *
from utils.ARGUS_UNet_train_test import *
from utils.ARGUS_visualization import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="ARGUS framework for PTX, PNB and ONSD tasks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpuid",type=int, default=0,help="GPU Device ID")
    parser.add_argument("--runid", type=int, default=0, help="Experiment Run ID")
    args = vars(parser.parse_args())
    
    device_num = args["gpuid"]
    run_id = args["runid"]

    img_folder = "/data/barry.ravichandran/repos/AnatomicRecon-POCUS-AI/ONSD/Data_ONSD/Preprocessed/Resized/"
    annotations_folder = "/data/barry.ravichandran/repos/AnatomicRecon-POCUS-AI/ONSD/Data_ONSD/Annotations/CLEANED/Resized/"
    
    p_prefix = [
        ' 158',
        ' 64',
        ' 139',
        ' 87',
        ' 117',
        ' 130',
        ' 205',
        ' 209',
        ' 213'
    ]
    n_prefix = [
        ' 23',
        ' 108',
        ' 119',
        ' 129',
        ' 163',
        ' 191',
        ' 228',
        ' 250'
    ]
   
    myNetwork = ARUNet_Nerve_Network()

    
    setup_vfold_files(myNetwork,
                    img_dir=img_folder,
                    anno_dir=annotations_folder,
                    p_prefix=p_prefix,
                    n_prefix=n_prefix)

    for i in range(myNetwork.num_folds):
        setup_training_vfold(myNetwork, 
                            vfold_num=i)
        train_vfold(myNetwork, 
                    run_id=run_id, 
                    device_num=device_num)

if __name__ == "__main__":
    main()