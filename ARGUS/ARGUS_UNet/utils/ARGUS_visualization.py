import numpy as np
import matplotlib.pyplot as plt
import os
import itk
from itk import TubeTK as tube
import torch

def view_training_image(self, image_num=0):
    img_name = self.all_train_images[image_num]
    print(img_name)
    img = itk.imread(img_name)
    lbl = itk.imread(self.all_train_labels[image_num])
    num_plots = 5
    num_slices = img.shape[0]
    step_slices = num_slices / num_plots
    plt.figure(figsize=[20, 10])
    for s in range(num_plots):
        slice_num = int(step_slices * s)
        plt.subplot(2, num_plots, s + 1)
        plt.imshow(img[slice_num, :, :])
        plt.subplot(2, num_plots, num_plots + s + 1)
        plt.imshow(lbl[slice_num, :, :])

def view_training_vfold_batch(self, batch_num=0):
    with torch.no_grad():
        for count, batch_data in enumerate(self.train_loader):
            if count == batch_num:
                inputs, labels = (batch_data["image"], batch_data["label"])
                num_images = inputs.shape[0]
                plt.figure(figsize=[30, 30])
                for i in range(num_images):
                    img = inputs[i]
                    lbl = labels[i]
                    num_channels = img.shape[0]
                    for c in range(num_channels):
                        plt.subplot(
                            num_images,
                            num_channels + 1,
                            i * (num_channels + 1) + c + 1,
                        )
                        plt.imshow(img[c, :, :])
                    plt.subplot(
                        num_images,
                        num_channels + 1,
                        i * (num_channels + 1) + num_channels + 1,
                    )
                    plt.imshow(lbl[0, :, :])
                break

def view_metric_curves(self, vfold_num, run_id=0):
    model_filename_base = (
        "./Results/"
        + self.filename_base
        + "-"
        + str(self.num_slices)
        + "s-VFold-Run"
        + str(run_id)
        + "/"
    )
    loss_file = model_filename_base + "loss_" + str(vfold_num) + ".npy"
    if os.path.exists(loss_file):
        epoch_loss_values = np.load(loss_file)

        metric_file = model_filename_base + "val_dice_" + str(vfold_num) + ".npy"
        metric_values = np.load(metric_file)

        plt.figure("Train", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.ylim([0.0, 0.8])

        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [2 * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.ylim([0.0, 0.8])

        plt.show()

def view_testing_results_vfold(self, model_type="best", run_id=[0], device_num=0):
    print("VFOLD =", self.vfold_num, "of", self.num_folds - 1)

    test_outputs = [self.test_vfold(model_type, r, device_num) for r in run_id]

    num_runs = len(run_id)

    artery_prior = 1
    class_artery = 1

    artery_min_size = 20000
    artery_max_size = 30000

    with torch.no_grad():
        for image_num, test_data in enumerate(self.test_loader):
            fname = os.path.basename(
                self.test_files[self.vfold_num][image_num]["image"]
            )
            print("Image:", fname)

            plt.figure("check", (18, 6))
            num_subplots = (
                max(self.net_in_channels, num_runs * self.num_classes) + 3
            )  # 3 = blank + ensemble(needle) + pred
            subplot_num = 1
            for c in range(self.net_in_channels):
                plt.subplot(2, num_subplots, subplot_num)
                plt.title(f"image")
                tmpV = test_data["image"][0, c, :, :]
                plt.imshow(tmpV, cmap="gray")
                subplot_num += 1
            plt.subplot(2, num_subplots, num_subplots)
            plt.title(f"label")
            tmpV = test_data["label"][0, 0, :, :]
            for c in range(self.num_classes):
                tmpV[0, c] = c
            plt.imshow(tmpV)
            subplot_num += 1

            # Indent by one plot
            subplot_num = num_subplots + 1
            prob_shape = test_outputs[0].shape
            prob_total = np.zeros(prob_shape)
            for run_num in range(num_runs):
                prob = np.empty(prob_shape)
                run_output = test_outputs[run_num]
                for c in range(self.num_classes):
                    itkProb = itk.GetImageFromArray(run_output[c, :, :])
                    imMathProb = tube.ImageMath.New(itkProb)
                    imMathProb.Blur(5)
                    itkProb = imMathProb.GetOutput()
                    prob[c] = itk.GetArrayFromImage(itkProb)
                pmin = prob.min()
                pmax = prob.max()
                prange = pmax - pmin
                prob = (prob - pmin) / prange
                prob[class_artery] = prob[class_artery] * artery_prior
                class_array = np.argmax(prob, axis=0)
                done = False
                while not done:
                    done = True
                    count = np.count_nonzero(class_array > 0)
                    while count < artery_min_size:
                        prob[class_artery] = prob[class_artery] * 1.05
                        class_array = np.argmax(prob, axis=0)
                        count = np.count_nonzero(class_array == class_artery)
                        done = False
                    while count > artery_max_size:
                        prob[class_artery] = prob[class_artery] * 0.95
                        class_array = np.argmax(prob, axis=0)
                        count = np.count_nonzero(class_array == class_artery)
                        done = False
                denom = np.sum(prob, axis=0)
                denom = np.where(denom == 0, 1, denom)
                prob = [prob[c] / denom for c in range(self.num_classes)]
                prob_total += prob
                for c in range(self.num_classes):
                    plt.subplot(2, num_subplots, subplot_num)
                    plt.title(f"Class " + str(c))
                    tmpV = prob[c, :, :]
                    plt.imshow(tmpV, cmap="gray")
                    subplot_num += 1

            prob_total = prob_total / num_runs
            subplot_num = num_subplots * 2 - 1
            plt.subplot(2, num_subplots, subplot_num)
            plt.title(f"Ensemble")
            tmpV = prob_total[class_artery, :, :]
            plt.imshow(tmpV, cmap="gray")
            subplot_num += 1

            class_array = np.argmax(prob_total, axis=0)
            class_image = itk.GetImageFromArray(class_array.astype(np.float32))
            imMathClassCleanup = tube.ImageMath.New(class_image)
            imMathClassCleanup.Erode(5, class_artery, 0)
            imMathClassCleanup.Dilate(5, class_artery, 0)
            class_image = imMathClassCleanup.GetOutputUChar()

            imMathClassCleanup.Threshold(class_artery, class_artery, 1, 0)
            class_artery_image = imMathClassCleanup.GetOutputUChar()

            seg = itk.itkARGUS.SegmentConnectedComponents.New(
                Input=class_artery_image
            )
            seg.SetKeepOnlyLargestComponent(True)
            seg.Update()
            class_artery_image = seg.GetOutput()
            class_artery_array = itk.GetArrayFromImage(class_artery_image)

            class_array = np.where(class_array == class_artery, 0, class_array)
            class_array = np.where(
                class_artery_array == 1, class_artery, class_array
            )
            plt.subplot(2, num_subplots, subplot_num)
            plt.title(f"Artery")
            tmpV = class_artery_array
            for c in range(self.num_classes):
                tmpV[0, c] = c
            plt.imshow(class_artery_array, cmap="gray")
            plt.show()