import torch
from torch import nn
from monai.networks.blocks import Convolution, Upsample


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        img_size: int,
        kernel_size: int = 9,
        patch_size: int = 7,
        n_classes: int = 1000,
        norm: bool = False,
        classification: bool = False,
        segmentation: bool = True,
        upsample_mode: str = 'deconv', # 'deconv' or 'pixelshuffle',
        new_t_dim: bool = False, # Switch Temporal dims for channels - only valid if single channel (grayscale) video
        t_channel_last: bool = False, # if shifting temporal to last dim before final inter.  Only valid if `new_t_dim = True'
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.norm = norm
        self.classification = classification
        self.segmentation = segmentation
        self.new_t_dim = new_t_dim
        self.t_channel_last = t_channel_last
        self.verbose = verbose

        if self.new_t_dim:
            dimensions = len(img_size) - 2
        else:
            dimensions = len(img_size) - 1

        if self.verbose:
            print(f'Patches Dimension: {dimensions}')

        if self.new_t_dim:
            self.patch_embedding  = Convolution(
                dimensions=dimensions,
                in_channels=img_size[3],
                out_channels=hidden_dim,
                kernel_size=(patch_size, patch_size),
                strides=(patch_size, patch_size),
                act='GELU',
                norm='BATCH',
                padding='valid'
            )
        else:
            self.patch_embedding  = Convolution(
                dimensions=dimensions,
                in_channels=img_size[0],
                out_channels=hidden_dim,
                kernel_size=(patch_size, patch_size, 1),
                strides=(patch_size, patch_size, 1),
                act='GELU',
                norm='BATCH',
                padding='valid'
            )
        blocks = []
        for i in range(depth):
            if self.new_t_dim:
                depthwise = Residual(
                                Convolution(
                                    dimensions=dimensions,
                                    in_channels=hidden_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=kernel_size,
                                    strides=1,
                                    groups=hidden_dim,
                                    act='GELU',
                                    norm='BATCH',
                                    padding='same',
                                    conv_only=True
                                )
                            )
            else:
                depthwise = Residual(
                    nn.Sequential(
                        Convolution(
                            dimensions=dimensions,
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=(kernel_size, kernel_size, 1),
                            strides=(1, 1, 1),
                            groups=hidden_dim,
                            act='GELU',
                            norm='BATCH',
                            padding='same',
                            conv_only=True
                        ),
                        Convolution(
                            dimensions=dimensions,
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=(1, 1, kernel_size),
                            strides=(1, 1, 1),
                            act='GELU',
                            norm='BATCH',
                            padding='same',
                        )
                    )
                )
            pointwise = Convolution(
                    dimensions=dimensions,
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    strides=1,
                    act='GELU',
                    norm='BATCH',
                    padding='same'
                    )

            blocks.append(nn.Sequential(depthwise, pointwise))

        self.blocks = nn.ModuleList(blocks)

        if self.norm:
            self.norm = nn.AdaptiveAvgPool2d((1, 1))
        if self.classification:
            self.classification_head = nn.Linear(hidden_dim, n_classes)
        if self.segmentation:
            # First upsample using either deconv or pixelshuffel then interpolate to exact same size

            # If need to create a new Temporal dimension
            if self.new_t_dim:
                # Interpolate last t channel (since nothing there before)
                if self.t_channel_last:
                    up_layer = Upsample(
                        spatial_dims=dimensions + 1,
                        in_channels=1,
                        out_channels=n_classes,
                        scale_factor=[patch_size, patch_size, 1],
                        mode=upsample_mode,
                        pre_conv=None
                    )
                else:
                    up_layer = Upsample(
                        spatial_dims=dimensions + 1,
                        in_channels=hidden_dim,
                        out_channels=n_classes,
                        scale_factor=[patch_size, patch_size, img_size[0]],
                        mode=upsample_mode,
                        pre_conv=None
                    )
                interp_layer = Upsample(
                    spatial_dims=dimensions + 1,
                    in_channels=n_classes,
                    out_channels=n_classes,
                    size=img_size[1:],
                    mode='nontrainable',
                    interp_mode='bilinear'
                )
            else:
                # if already doing 3D convolution
                up_layer = Upsample(
                    spatial_dims=dimensions,
                    in_channels=hidden_dim,
                    out_channels=n_classes,
                    scale_factor=[patch_size, patch_size, 1],
                    mode=upsample_mode,
                    pre_conv=None
                )
                interp_layer = Upsample(
                    spatial_dims=dimensions,
                    in_channels=n_classes,
                    out_channels=n_classes,
                    size=img_size[1:],
                    mode='nontrainable',
                    interp_mode='bilinear'
                )

            self.segmentation_head = nn.Sequential(up_layer, interp_layer)

    def forward(self, x):
        if self.verbose:
            print(f'Initial Shape {x.shape}')
        if self.new_t_dim:
            x = torch.transpose(torch.squeeze(x), 1, 3)
            if self.verbose:
                print(f'Squeezed Shape {x.shape}')
        x = self.patch_embedding(x)
        if self.verbose:
            print(f'After embedding Shape {x.shape}')
        hidden_states_out = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            hidden_states_out.append(x)
            if self.verbose:
                print(f'Shape after layer {i}: {x.shape}')
        if self.norm:
            x = self.norm(x)
        if self.classification:
            x = self.classification_head(x)
        if self.segmentation:
            if self.new_t_dim:
                if self.t_channel_last:
                    x = torch.unsqueeze(torch.transpose(x, 1, 3), 1)
                else:
                    x = torch.unsqueeze(x, -1)
            if self.verbose:
                print(f'Shape before segmentations: {x.shape}')

            x = self.segmentation_head(x)
        if self.verbose:
            print(f'Output Shape {x.shape}')
        return x
