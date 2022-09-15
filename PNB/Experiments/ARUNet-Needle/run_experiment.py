from torch.multiprocessing import freeze_support
from ARGUS_Needle_Experiment import train_parallel_runs

def run_experiment():
    img_dir = "../../Data_PNB/stephen_prep"
    anno_dir = "../../Data_PNB/stephen_prep"

    do_needle = True
    runs = [0,1]
    vfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    devices = [0,1]

    train_parallel_runs(img_dir, anno_dir, runs, vfolds, devices, do_needle)

if __name__ == '__main__':
    freeze_support()
    run_experiment()
