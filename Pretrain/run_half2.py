import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_train import ARGUS_segmentation_train

from monai.utils.misc import set_determinism
set_determinism()

nnet = ARGUS_segmentation_train("ARGUS_pretrain_half_ar.cfg", "vfold", device_num=2)
nnet.setup_vfold_files()

set_determinism(seed=None)

run_num = 2
for vfn in range(3,nnet.num_folds):
    nnet.init_model(run_num)
    nnet.setup_training_vfold(vfn)
    nnet.train_vfold(run_num)
