import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_train import ARGUS_segmentation_train

nnet = ARGUS_segmentation_train("ARGUS_pretrain_half_ar.cfg", "vfold", device_num=0)
nnet.setup_vfold_files()

run_num = 2
for vfn in range(nnet.num_folds):
    nnet.setup_training_vfold(vfn)
    nnet.train_vfold(run_num)
