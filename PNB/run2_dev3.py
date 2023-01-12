import site
site.addsitedir("../ARGUS")

from ARGUS_pnb_ar_train import ARGUS_pnb_ar_train

run_num = 2
device_num = 3
best_model_num = 4
nnet = ARGUS_pnb_ar_train("../ARGUS/ARGUS_pnb_ar.cfg", "vfold", device_num=device_num)
nnet.setup_vfold_files()
for vfn in range(nnet.num_folds):
    nnet.setup_training_vfold(vfn, run_num)
    pretrained_model_name = "./Pretrained_Models/pretrain_half_vfold_run"+str(run_num)+"/best_model_"+str(best_model_num)+".pth"
    nnet.load_model(run_num, pretrained_model_name)
    nnet.train_vfold(run_num)
