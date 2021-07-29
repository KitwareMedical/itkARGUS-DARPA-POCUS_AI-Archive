from pytorch_unet import *
import torch
from monai.losses import DiceLoss

pred = torch.zeros((5,1,128,128))
pred = torch.cat((pred, torch.tensor(0.9) * torch.ones((5,1,128,128))), dim=1)
target = torch.ones((5,1,128,128))
print(pred[0,:,0,0])
print(target[0,:,0,0])

print(pred.size(), target.size())

diceloss = DiceLoss(to_onehot_y=True)
dice = diceloss(pred, target)
print("should be 1.0")
print("dice:", dice)
