import site
site.addsitedir("../ARGUS")

from ARGUS_ptx_ar_train import ARGUS_ptx_ar_train

for run_num in range(1):
    nnet = ARGUS_ptx_ar_train("../ARGUS/ARGUS_ptx_ar.cfg", "vfold", device_num=0)
    nnet.setup_vfold_files()
    for vfn in range(nnet.num_folds):
        nnet.setup_training_vfold(vfn)
        pretrained_model_name = "./Pretrained_Models/pretrain_ptx_final_run"+str(run_num)+"/best_model_0.pth"
        nnet.load_model(run_num, pretrained_model_name)
        nnet.train_vfold(run_num)
