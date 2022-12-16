from ARGUS_Network import ARGUS_Network

nnet = ARGUS_Network("PreTrainPTX")
nnet.setup_vfold_files()

nnet.setup_training_vfold(0)

nnet.max_epochs = 5000
run_num = 2
device_num = 2
nnet.train_vfold(run_num,device_num)
