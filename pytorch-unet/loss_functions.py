import torch
import torch.nn.functional as F

"""
This file will contain loss functions:
 - weighted multicategorical crossentropy with bias regularization
 - another with same with special kernel regularization
 - and a dice coefficient function
"""
def dice_loss(pred, target, smooth=1.0):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # pred is logits, target is not one-hot
    print(pred.size(), target.size())
    # Need to change target to one-hot

    num_classes = pred.size(1)
    print("num classes", num_classes)
    target[:,0,:,:] = F.one_hot(target[:,0,:,:], num_classes=num_classes)

    # have to use contiguous since they may from a torch.view op
    print(pred.size(), target.size())
    assert False
    pflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (pflat * tflat).sum()

    A_sum = torch.sum(pflat * pflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1.0 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth) )

def WCE_bias_regularization(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)

        return loss

    return loss