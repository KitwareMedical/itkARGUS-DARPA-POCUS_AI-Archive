import site
site.addsitedir("../ARGUS")

from ARGUS_ptx_ar_train import ARGUS_ptx_ar_train

nnet = ARGUS_ptx_ar_train("../ARGUS/ARGUS_ptx_ar.cfg", "vfold", device_num=1)
nnet.image_dirname = ["Data_PTX/images"]
nnet.label_dirname = ["Data_PTX/labels"]
num_folds = nnet.num_folds
nnet.num_folds = 1
nnet.randomize_folds = False
nnet.train_data_portion = 1.0
nnet.validation_data_portion = 1.0
nnet.test_data_portion = 1.0

nnet.setup_vfold_files()

nnet.setup_testing_vfold(vfold_num=0)

for r in range(nnet.num_models):
    print("*** Run", r)
    for vf in range(num_folds):
        test_output = nnet.view_testing_results_vfold(model_type="best", run_ids=[r], model_vfold=vf, summary_only=True)
