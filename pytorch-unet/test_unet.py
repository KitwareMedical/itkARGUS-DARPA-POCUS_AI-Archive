from pytorch_unet import *
from torch import nn

model = UNet(128,2)

# print(list(model.parameters()))
# print(model.down_layers[0])
# print(list(model.down_layers[0].conv.parameters())[1])

params = model.get_parameters()

# print(list(model.down_layers[0].parameters()))
# print(params[2])

x = torch.randn((5,1,128,128))
pred = model(x)
label = torch.rand_like(pred).round().int()
label = torch.argmax(label, dim=1)
class_weights = torch.tensor([0.5, 1.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
loss = criterion(pred, label)
loss.backward()

optim = torch.optim.SGD(params, lr=1e-2, momentum=0.9)
optim.step()