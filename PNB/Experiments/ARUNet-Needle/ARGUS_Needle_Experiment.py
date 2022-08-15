import os
import sys
import torch.multiprocessing.spawn as spawn

from ARGUS_Needle_Network import ARGUS_Needle_Network


def run_train_process(proc_id, nnet, vfold_num, run_num, devices):
    print("      ---Starting run", run_num+proc_id)
    std_out_filename = (
        "ARGUS_Needle_Network-vf"
        + str(vfold_num)
        + "-r"
        + str(run_num + proc_id)
        + ".out.txt"
    )
    sys.stdout = open(std_out_filename, "w")
    nnet.train_vfold(run_num + proc_id, devices[proc_id])
    sys.stdout.close()


def train_parallel_runs(
    img_dir, anno_dir, run_ids=[0], vfold_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], devices=[0, 1]
):
    nnet = ARGUS_Needle_Network()
    nnet.setup_vfold_files(img_dir, anno_dir)

    num_runs = len(run_ids)

    for vfold_num in vfold_ids:
        print("*****Starting vFold = ", vfold_num)
        nnet.setup_training_vfold(vfold_num)
        for run_num in range(0, num_runs, len(devices)):
            if run_num + len(devices) < num_runs:
                print("   *****Starting runs", run_ids[run_num], "-", run_ids[run_num + len(devices)])
                spawn(
                    run_train_process,
                    (nnet, vfold_num, run_ids[run_num], devices),
                    nprocs=len(devices),
                    join=True,
                    daemon=False,
                )
            else:
                num_procs = num_runs - run_num
                print(
                    "   *****Starting final runs", run_ids[run_num], "-", run_ids[run_num + num_procs - 1]
                )
                spawn(
                    run_train_process,
                    (nnet, vfold_num, run_ids[run_num], devices),
                    nprocs=num_procs,
                    join=True,
                    daemon=False,
                )
