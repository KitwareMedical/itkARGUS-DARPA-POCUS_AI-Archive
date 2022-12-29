from ARGUS_train_ptx_ar import ARGUS_train_ptx_ar

nnet = ARGUS_train_ptx_ar("vfold", device_num=0)

nnet.setup_vfold_files()

for run_num in range(3):
    for vfn in range(nnet.num_folds):
        nnet.setup_training_vfold(vfn)
        nnet.train_vfold(run_num)
