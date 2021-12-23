def arnet_load_model(filename, device):
    num_classes = 3

    net_in_dims = 3
    net_in_channels = 1
    net_channels=(16, 32, 64, 128, 32)
    net_strides=(2, 2, 2, 2)

    model = UNet(
        dimensions=net_in_dims,
        in_channels=net_in_channels,
        out_channels=num_classes,
        channels=net_channels,
        strides=net_strides,
        num_res_units=2,
        norm=Norm.BATCH,
        ).to(device)    
    model.load_state_dict(torch.load(filename))
    model.eval()

    return model

def arnet_preprocess_data(input_array):
    num_slices = 48

    arnet_input_array = np.empty([1, 1,
                           input_array.shape[0], input_array.shape[1],
                           num_slices])

    Scale = ScaleIntensityRange(
        a_min=0, a_max=255,
        b_min=0.0, b_max=1.0)
    input_array_scaled = Scale(input_array)

    Crop = ARGUS_RandSpatialCropSlices(
        num_slices=num_slices,
        center_slice=30,
        axis=2)
    arnet_input_array[0, 0] = Crop(input_array_scaled)

    arnet_input_tensor = ToTensor(arnet_input_array.astype(np.float32))

    return arnet_input_tensor

def arnet_inference(arnet_input_tensor, arnet_model, device):
    num_slices = 48

    size_x = 320
    size_y = 320
    roi_size = (size_x,size_y,num_slices)

    pleura_prior = 1

    class_pleura = 1
    class_rib = 2

    min_size = 110000
    max_size = 160000

    with torch.no_grad():
        test_outputs = sliding_window_inference(
            arnet_input_tensor.to(device), roi_size, 1, model)
        prob_shape = test_outputs[0,:,:,:,:].shape
        prob = np.empty(prob_shape)
        for c in range(num_classes):
            itkProb = itk.GetImageFromArray(test_outputs[0,c,:,:,:].cpu())
            imMathProb = ttk.ImageMath.New(itkProb)
            imMathProb.Blur(5)
            itkProb = imMathProb.GetOutput()
            prob[c] = itk.GetArrayFromImage(itkProb)
        class_array = np.zeros(prob[0].shape)
        pmin = prob[0].min()
        pmax = prob[0].max()
        for c in range(1,num_classes):
            pmin = min(pmin, prob[c].min())
            pmax = max(pmax, prob[c].max())
        prange = pmax - pmin
        prob = (prob - pmin) / prange
        prob[class_pleura] = prob[class_pleura] * pleura_prior
        done = False
        while not done:
            done = True
            count = np.count_nonzero(class_array>0)
            prior_factor = 1
            while count<min_size:
                prior_factor *= 1.05
                prob[class_pleura] = prob[class_pleura] * 1.05
                prob[class_rib] = prob[class_rib] * 1.05
                class_array = np.argmax(prob,axis=0)
                count = np.count_nonzero(class_array>0)
                done = False
            while count>max_size:
                prior_factor *= 0.95
                prob[class_pleura] = prob[class_pleura] * 0.95
                prob[class_rib] = prob[class_rib] * 0.95
                class_array = np.argmax(prob,axis=0)
                count = np.count_nonzero(class_array>0)
                done = False

        class_array = np.where(class_array==1,1,0)
        class_image = itk.GetImageFromArray(class_array.astype(np.float32))
        imMathClassCleanup = ttk.ImageMath.New(class_image)
        imMathClassCleanup.Erode(5,class_pleura,0)
        imMathClassCleanup.Dilate(5,class_pleura,0)
        class_output = imMathClassCleanup.GetOutputUChar()
        
        itkSegmentConnectedComponents =
            itk.itkARGUS.SegmentConnectedComponents
        seg = itkSegmentConnectedComponents.New(Input=class_output)
        seg.SetKeepOnlyLargestComponent(True)
        seg.Update()

        class_output = seg.GetOutput()
        class_output_array = itk.GetArrayFromImage(class_output)

        return class_output_array
