import site
site.addsitedir("../ARGUS")

from ARGUS_ett_roi_train import ARGUS_ett_roi_train

nnet = ARGUS_ett_roi_train("../ARGUS/ARGUS_ett_roi.cfg", "vfold", device_num=1)
num_folds = nnet.num_folds
nnet.use_persistent_cache = False
nnet.num_folds = 1
nnet.randomize_folds = False
nnet.train_data_portion = 1.0
nnet.validation_data_portion = 1.0
nnet.test_data_portion = 1.0

nnet.setup_vfold_files()

for r in range(nnet.num_models):
    print("*** Run", r)
    nnet.setup_testing_vfold(vfold_num=0, run_num=r)
    for vf in range(num_folds):
        test_output = nnet.view_testing_results_vfold(model_type="best", run_ids=[r], model_vfold=vf, summary_only=True)
