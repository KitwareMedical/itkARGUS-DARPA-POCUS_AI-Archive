import site
site.addsitedir("../ARGUS")

from ARGUS_segmentation_train import ARGUS_segmentation_train

nnet = ARGUS_segmentation_train("ARGUS_pretrain_half_ar.cfg", "vfold", device_num=0)
nnet.image_dirname = ["Data_Pretrain/images_ptx"]
nnet.label_dirname = ["Data_Pretrain/labels_ptx"]
nnet.num_folds = 1
nnet.randomize_folds = False
nnet.train_data_portion = 1.0
nnet.validation_data_portion = 1.0
nnet.test_data_portion = 1.0

nnet.setup_vfold_files()

nnet.setup_testing_vfold(vfold_num=0)

for vf in range(5):
    test_output = nnet.view_testing_results_vfold(model_type="best", run_ids=[0], model_vfold=vf, summary_only=True)
