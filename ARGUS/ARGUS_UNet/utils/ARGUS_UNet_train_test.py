import torch
import os
import numpy as np
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

def train_vfold(self, run_id=0, device_num=0):
    model_filename_base = (
        "./Results/"
        + self.filename_base
        + "-"
        + str(self.num_slices)
        + "s-VFold-Run"
        + str(run_id)
    )
    if not os.path.exists(model_filename_base):
        os.makedirs(model_filename_base)
    model_filename_base = model_filename_base + "/"

    device = torch.device("cuda:" + str(device_num))

    model = UNet(
        dimensions=self.net_dims,
        in_channels=self.net_in_channels,
        out_channels=self.num_classes,
        channels=self.net_channels,
        strides=self.net_strides,
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    val_interval = 2

    post_pred = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True, to_onehot=True, num_classes=self.num_classes),
        ]
    )
    post_label = Compose(
        [EnsureType(), AsDiscrete(to_onehot=True, num_classes=self.num_classes)]
    )

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(self.max_epochs):
        print("-" * 10)
        print(f"{self.vfold_num}: epoch {epoch + 1}/{self.max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in self.train_loader:
            step += 1
            inputs, labels = (batch_data["image"], batch_data["label"])
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step} /"
                f" {len(self.train_files[self.vfold_num])//self.train_loader.batch_size},"
                f" train_loss: {loss.item():.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(
            f"{self.vfold_num} epoch {epoch+1}" f" average loss: {epoch_loss:.4f}"
        )

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in self.val_loader:
                    val_inputs, val_labels = (val_data["image"], val_data["label"])
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    roi_size = (self.size_x, self.size_y)
                    sw_batch_size = self.batch_size_val
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                    )
                    # val_outputs = model(val_inputs)
                    val_outputs = [
                        post_pred(i) for i in decollate_batch(val_outputs)
                    ]
                    val_labels = [
                        post_label(i) for i in decollate_batch(val_labels)
                    ]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if epoch > 100:
                    metric = (
                        metric_values[-1] + metric_values[-2] + metric_values[-3]
                    ) / 3
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(
                            model.state_dict(),
                            model_filename_base
                            + "best_model.vfold_"
                            + str(self.vfold_num)
                            + ".pth",
                        )
                        print("saved new best metric model")
                print(
                    f"Current epoch: {epoch + 1}"
                    f" current mean dice: {metric:.4f}"
                )
                print(
                    f"Best mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                torch.save(
                    model.state_dict(),
                    model_filename_base
                    + "last_model.vfold_"
                    + str(self.vfold_num)
                    + ".pth",
                )
                np.save(
                    model_filename_base + "loss_" + str(self.vfold_num) + ".npy",
                    epoch_loss_values,
                )
                np.save(
                    model_filename_base
                    + "val_dice_"
                    + str(self.vfold_num)
                    + ".npy",
                    metric_values,
                )

def test_vfold(self, model_type="best", run_id=0, device_num=0):
    model_filename_base = (
        "./Results/"
        + self.filename_base
        + "-"
        + str(self.num_slices)
        + "s-VFold-Run"
        + str(run_id)
        + "/"
    )

    model_file = (
        model_filename_base
        + model_type
        + "_model.vfold_"
        + str(self.vfold_num)
        + ".pth"
    )

    device = torch.device("cuda:" + str(device_num))

    test_outputs_total = []

    if os.path.exists(model_file):
        model = UNet(
            dimensions=self.net_dims,
            in_channels=self.net_in_channels,
            out_channels=self.num_classes,
            channels=self.net_channels,
            strides=self.net_strides,
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

        model.load_state_dict(torch.load(model_file))
        model.eval()

        with torch.no_grad():
            for b, test_data in enumerate(self.test_loader):
                roi_size = (self.size_x, self.size_y)
                test_outputs = sliding_window_inference(
                    test_data["image"].to(device),
                    roi_size,
                    self.batch_size_test,
                    model,
                ).cpu()
                if b == 0:
                    test_outputs_total = test_outputs
                else:
                    test_outputs_total = np.concatenate(
                        (test_outputs_total, test_outputs), axis=0
                    )

    return test_outputs_total