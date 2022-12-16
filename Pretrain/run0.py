from arnet import arnet

nnet = arnet("Pretrain")
nnet.setup_vfold_files()

nnet.setup_training_vfold(0)

run_num = 0
device_num = 0
nnet.train_vfold(run_num,device_num)
