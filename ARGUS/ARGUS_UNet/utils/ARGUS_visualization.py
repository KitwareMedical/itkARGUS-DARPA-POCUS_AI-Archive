import numpy as np
import matplotlib.pyplot as plt
import os
import itk
from itk import TubeTK as tube
import torch
from scipy.ndimage import rotate

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
                        plt.imshow(rotate(img[c, :, :],270))
                    plt.subplot(
                        num_images,
                        num_channels + 1,
                        i * (num_channels + 1) + num_channels + 1,
                    )
                    plt.imshow(rotate(lbl[0, :, :],270))
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

def view_testing_results_vfold(self, test_outputs, test_images, test_labels, model_type="best", run_id=[0], device_num=0):
    print("VFOLD =", self.vfold_num, "of", self.num_folds - 1)
    
    num_runs = len(run_id)

    class_prior = np.ones(self.num_classes)

    if self.target_data == "ONSD":
        target_class = self.class_nerve
    else:
        target_class = self.class_artery

    with torch.no_grad():
        for img in range(0,test_outputs[0].shape[0]):
            fname = os.path.basename(
                self.test_files[self.vfold_num][img]["image"]
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
                tmpV = test_images[0][img, c, :, :]
                plt.axis('off')
                plt.imshow(rotate(tmpV,270), cmap="gray")
                subplot_num += 1
            plt.subplot(2, num_subplots, num_subplots)
            plt.title(f"label")
            tmpV = test_labels[0][img, 0, :, :]
            for c in range(self.num_classes):
                tmpV[0, c] = c
            plt.axis('off')
            plt.imshow(rotate(tmpV,270))
            subplot_num += 1

            # Indent by one plot
            subplot_num = num_subplots + 1
            prob_shape = test_outputs[0][img].shape
            prob_total = np.zeros(prob_shape)
            for run_num in range(num_runs):
                prob = np.empty(prob_shape)
                run_output = test_outputs[run_num]
                for c in range(self.num_classes):
                    itkProb = itk.GetImageFromArray(run_output[img,c, :, :])
                    imMathProb = tube.ImageMath.New(itkProb)
                    imMathProb.Blur(5)
                    itkProb = imMathProb.GetOutput()
                    prob[c] = itk.GetArrayFromImage(itkProb)
                pmin = prob.min()
                pmax = prob.max()
                prange = pmax - pmin
                prob = (prob - pmin) / prange
                prob[target_class] = prob[target_class] * class_prior[target_class]
                class_array = np.argmax(prob, axis=0)
                done = False
                while not done:
                    done = True
                    count = np.count_nonzero(class_array > 0)
                    while count < self.class_min_size[target_class]:
                        prob[target_class] = prob[target_class] * 1.05
                        class_array = np.argmax(prob, axis=0)
                        count = np.count_nonzero(class_array == target_class)
                        done = False
                    while count > self.class_max_size[target_class]:
                        prob[target_class] = prob[target_class] * 0.95
                        class_array = np.argmax(prob, axis=0)
                        count = np.count_nonzero(class_array == target_class)
                        done = False
                denom = np.sum(prob, axis=0)
                denom = np.where(denom == 0, 1, denom)
                prob = [prob[c] / denom for c in range(self.num_classes)]
                prob_total += prob
                for c in range(self.num_classes):
                    plt.subplot(2, num_subplots, subplot_num)
                    plt.title(f"Class " + str(c))
                    tmpV = prob[c]
                    plt.axis('off')
                    plt.imshow(rotate(tmpV,270), cmap="gray")
                    subplot_num += 1

            prob_total = prob_total / num_runs
            subplot_num = num_subplots * 2 - 1
            plt.subplot(2, num_subplots, subplot_num)
            plt.title(f"Ensemble")
            tmpV = prob_total[target_class, :, :]
            plt.axis('off')
            plt.imshow(rotate(tmpV,270), cmap="gray")
            subplot_num += 1

            class_array = np.argmax(prob_total, axis=0)
            class_image = itk.GetImageFromArray(class_array.astype(np.float32))
            imMathClassCleanup = tube.ImageMath.New(class_image)
            imMathClassCleanup.Erode(self.erosion_size, target_class, 0)
            imMathClassCleanup.Dilate(self.dilation_size, target_class, 0)
            class_image = imMathClassCleanup.GetOutputUChar()

            imMathClassCleanup.Threshold(target_class, target_class, 1, 0)
            self.target_class_image = imMathClassCleanup.GetOutputUChar()

            seg = itk.itkARGUS.SegmentConnectedComponents.New(
                Input=self.target_class_image
            )
            seg.SetKeepOnlyLargestComponent(True)
            seg.Update()
            self.target_class_image = seg.GetOutput()
            self.target_class_array = itk.GetArrayFromImage(self.target_class_image)

            class_array = np.where(class_array == target_class, 0, class_array)
            class_array = np.where(
                self.target_class_array == 1, target_class, class_array
            )
            plt.subplot(2, num_subplots, subplot_num)
            plt.title(f"Artery")
            tmpV = self.target_class_array
            for c in range(self.num_classes):
                tmpV[0, c] = c
            plt.axis('off')
            plt.imshow(rotate(self.target_class_array,270))
            plt.show()
            save_img = itk.GetImageFromArray(self.target_class_array)
            save_inference_results(self,
                                class_image=save_img,
                                fname=fname)

def save_inference_results(self,class_image,fname):
    if not os.path.exists(self.result_files_savepath):
        os.makedirs(self.result_files_savepath)
    itk.imwrite(class_image,os.path.join(self.result_files_savepath,fname.split('.nii.gz')[0] + '_result.nii.gz'),compression=True)