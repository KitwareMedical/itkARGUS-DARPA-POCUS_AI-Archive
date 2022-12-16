from arnet import arnet

nnet = arnet("Pretrain")
nnet.setup_vfold_files()

nnet.setup_training_vfold(0)

run_num = 1
device_num = 1
nnet.train_vfold(run_num,device_num)
