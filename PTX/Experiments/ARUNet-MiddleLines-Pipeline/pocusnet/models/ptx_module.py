from typing import Any, List

import torch
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
# from torchmetrics.classification.accuracy import Accuracy
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.networks.utils import one_hot
from copy import deepcopy


class PTXLitModule(LightningModule):
    """Example of LightningModule for PTX classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        num_classes: int = 3,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = DiceLoss(to_onehot_y=True, softmax=True)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        self.train_acc = DiceMetric(include_background=False, reduction="mean")
        self.val_acc = DiceMetric(include_background=False, reduction="mean")
        self.test_acc = DiceMetric(include_background=False, reduction="mean")

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def compute_monai_metric(self, y_pred, y, acc_metric):
        acc_metric(
            y_pred=one_hot(y_pred, self.hparams.num_classes),
            y=one_hot(y, self.hparams.num_classes)
        )
        return acc_metric.aggregate().item()

    def step(self, batch: Any):
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.compute_monai_metric(preds, targets, self.train_acc)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else
        # backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.compute_monai_metric(preds, targets, self.val_acc)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False)

        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(deepcopy(
                batch['image'][:, :, :, :, 2]), padding=10)
            self.logger.experiment[0].add_image('val/imgs', grid, global_step=self.current_epoch)

            grid = torchvision.utils.make_grid(deepcopy(
                preds[:, :, :, :, 2]).float(), normalize=True,
                value_range=(0, self.hparams.num_classes - 1), padding=10)
            self.logger.experiment[0].add_image('val/pred', grid, global_step=self.current_epoch)
            grid = torchvision.utils.make_grid(deepcopy(
                targets[:, :, :, :, 2]).float(), normalize=True,
                value_range=(0, self.hparams.num_classes - 1), padding=10)
            self.logger.experiment[0].add_image('val/gt', grid, global_step=self.current_epoch)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.aggregate().item()
        self.val_acc_best.update(acc)

        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            on_epoch=True,
            prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.compute_monai_metric(preds, targets, self.test_acc)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(deepcopy(batch['image'][:, :, :, :, 2]), padding=10)
            self.logger.experiment[0].add_image('test/imgs', grid)

            grid = torchvision.utils.make_grid(deepcopy(
                preds[:, :, :, :, 2]).float(), normalize=True,
                value_range=(0, self.hparams.num_classes - 1), padding=10)
            self.logger.experiment[0].add_image('test/pred', grid)
            grid = torchvision.utils.make_grid(deepcopy(
                targets[:, :, :, :, 2]).float(), normalize=True,
                value_range=(0, self.hparams.num_classes - 1), padding=10)
            self.logger.experiment[0].add_image('test/gt', grid)

        self.log("hp_metric", acc)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


def _demo():
    # import matplotlib.pyplot as plt
    # Import the net
    from pocusnet.models.components.unet import UNet
    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128),
        strides=(2, 2),
        num_res_units=2,
        norm="batch",
    )

    # Create the lit module
    import pocusnet.models.ptx_module as ptx_m
    ptx_model = ptx_m.PTXLitModule(net=net)

    from pocusnet.models.components.patches import ConvMixer
    image_shape = [1, 160, 320, 32]
    patches = ConvMixer(
            hidden_dim=124,
            depth=2,
            img_size=image_shape,
            n_classes=3,
            new_t_dim=True,
            t_channel_last=True,
            verbose=True
            )

    import pocusnet.models.ptx_module as ptx_m
    ptx_model = ptx_m.PTXLitModule(net=patches)

    # Get data to test
    from monai.utils import first
    import pocusnet.datamodules.ptx_datamodule as ptx_d
    d = ptx_d.PTXDataModule(
        data_dir="/data/krsdata2-pocus-ai-synced/root/Data_PTX/VFoldData/BAMC-PTX*Sliding-Annotations-Linear/")
    d.prepare_data()
    d.setup()
    train_dataloader = d.train_dataloader()
    check_data = first(train_dataloader)
    imgnum = 1
    image, label = (check_data["image"][imgnum][0],  # NOQA
                    check_data["label"][imgnum][0])
    print(f'Image batch size {check_data["image"].shape}')
    print(f'Image Shape {image.shape}')

    # Check forward pass
    loss, preds, targets = ptx_model.step(check_data)
    print(f'pred shape: {preds.shape} vs label shape {targets.shape}')

    # Check
    loss, preds, targets = ptx_model.training_step(
        batch=check_data, batch_idx=1)
    loss, preds, targets = ptx_model.validation_step(
        batch=check_data, batch_idx=1)
    ptx_model.validation_epoch_end([])
    loss, preds, targets = ptx_model.test_step(batch=check_data, batch_idx=1)


if __name__ == "__main__":
    _demo()
