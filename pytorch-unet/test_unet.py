from pytorch_unet import *
from torch import nn
from monai.losses import DiceLoss

model = UNet_rect_kernels(128,2)

# print(list(model.parameters()))
# print(model.down_layers[0])
# print(list(model.down_layers[0].conv.parameters())[1])

params = model.parameters()
named_params = model.named_parameters()
num = 0
for k, x in model.named_parameters():
    print(k)
    # print(x)
    if num >= 10: break
    num += 1

quit()

# print(list(model.down_layers[0].parameters()))
# print(params[2])

x = torch.randn((5,1,128,128))
pred = model(x)
label = torch.rand_like(pred).round().int()
label = torch.argmax(label, dim=1)
class_weights = torch.rand(2)
criterion = nn.CrossEntropyLoss(weight=class_weights)
loss = criterion(pred, label)
loss.backward()
diceloss = DiceLoss(to_onehot_y=True, softmax=True)
dice = diceloss(pred, label.view((5,1,128,128)))
print("dice:", dice)

optim = torch.optim.SGD(params, lr=1e-2, momentum=0.9)
optim.step()

print('testing finished sucessfully')
