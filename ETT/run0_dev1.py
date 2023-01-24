import site
site.addsitedir("../ARGUS")

from ARGUS_ett_roi_train import ARGUS_ett_roi_train

run_num = 0
device_num = 1
nnet = ARGUS_ett_roi_train("../ARGUS/ARGUS_ett_roi.cfg", "vfold", device_num)
nnet.setup_vfold_files()

for vfn in range(nnet.num_folds):
    nnet.init_model(run_num)
    nnet.setup_training_vfold(vfn,run_num)
    nnet.train_vfold(run_num)
