from torch.multiprocessing import freeze_support
from ARGUS_Needle_Experiment import train_parallel_runs

def run_experiment():
    img_dir = "../../Data_PNB/Preprocessed"
    anno_dir = "../../Data_PNB/annotations/Sean_May22_CLEANED"

    num_runs = 5
    vfolds = [0, 1, 2, 3, 4]
    devices = [0, 1, 2]

    train_parallel_runs(img_dir, anno_dir, num_runs, vfolds, devices)

if __name__ == '__main__':
    freeze_support()
    run_experiment()
