from ARGUS_taskid import ARGUS_taskid

nnet = ARGUS_taskid("vfold")
nnet.setup_vfold_files()

run_num = 0
device_num = 0
for vfn in range(nnet.num_folds):
    nnet.setup_training_vfold(vfn)
    nnet.train_vfold(run_num,device_num)
