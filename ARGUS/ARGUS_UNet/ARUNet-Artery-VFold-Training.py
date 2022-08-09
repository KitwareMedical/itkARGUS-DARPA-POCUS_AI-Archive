from experiments.PNB.ARUNet_Artery_Network import ARUNet_Artery_Network
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

    img_folder = "/data/barry.ravichandran/repos/AnatomicRecon-POCUS-AI/PNB/Data_PNB/annotations_yuri/"
    annotations_folder = "/data/barry.ravichandran/repos/AnatomicRecon-POCUS-AI/PNB/Data_PNB/annotations_yuri/CLEANED/"
    
    p_prefix = [' 11',
                ' 67', 
                ' 93', 
                ' 94', 
                ' 134', 
                ' 211', 
                ' 222A',  
                ' 153', 
                ' 240',  
                ' 193']
    n_prefix = [' 57',
            ' 136', 
            ' 179', 
            ' 189', 
            ' 204', 
            ' 205', 
            ' 217', 
            ' 238',  
            ' 39',
            ' 191']
   
    myNetwork = ARUNet_Artery_Network()

    setup_vfold_files(myNetwork,
                    img_dir=img_folder,
                    anno_dir=annotations_folder,
                    p_prefix=p_prefix,
                    n_prefix=n_prefix)

    setup_training_vfold(myNetwork, 
                        vfold_num=0)
    
    train_vfold(myNetwork, 
                run_id=run_id, 
                device_num=device_num)

if __name__ == "__main__":
    main()