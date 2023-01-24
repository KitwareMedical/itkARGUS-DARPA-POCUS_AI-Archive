import site
site.addsitedir("../ARGUS")

from ARGUS_ptx_roi_train import ARGUS_ptx_roi_train

#from monai.utils.misc import set_determinism
#set_determinism()

nnet = ARGUS_ptx_roi_train("../ARGUS/ARGUS_ptx_roi.cfg", "vfold", 3)
nnet.setup_vfold_files()

#set_determinism(seed=None)

for run_num in range(nnet.num_models):
    for vfn in range(nnet.num_folds):
        nnet.init_model(run_num)
        nnet.setup_training_vfold(vfn,run_num)
        nnet.train_vfold(run_num)
