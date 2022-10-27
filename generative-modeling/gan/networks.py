import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["PYTORCH_JIT"] = "0"

class UpSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.upscale_factor = upscale_factor
        self.upscale_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2d = nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, padding=padding)
        #print(self.args)
    
    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling.
        # 1. Duplicate x channel wise upscale_factor^2 times.
        # (N, C, H, W) = (_, 3, 32, 32)
        x = x.repeat(1, int(self.upscale_factor**2), 1, 1)   # (*, C*Up^2, H, W)
        # 2. Then re-arrange to form an image of shape (batch x channel x height*upscale_factor x width*upscale_factor).
        x = self.upscale_shuffle(x)   #  (*, C, H*Up, W*Up)
        # 3. Apply convolution.
        x = self.conv2d(x)   # (_, 128, 64, 64)
        # Hint for 2. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle
        return x 
    
class DownSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.downscale_ratio=downscale_ratio
        self.downscale_shuffle = nn.PixelUnshuffle(self.downscale_ratio)
        # 
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=kernel_size, padding=padding)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling.
        # Hint for 1. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle
        # 1. Re-arrange to form an image of shape: (batch x channel * upscale_factor^2 x height x width).
        x = x.repeat(1, int(self.downscale_ratio**2), 1, 1)   # (*, C*Up^2, H, W)
        # 2. Then split channel wise into upscale_factor^2 number of images of shape: (batch x channel x height x width).
        x = self.downscale_shuffle(x)  # (N, C*Up^2, H/Up, W/Up) = (*, 48, 16, 16)
        # 3. Average the images into one and apply convolution.
        x = torch.mean(x,dim=1).unsqueeze(1)
        x = self.conv2d(x)  # (*, n_filters, 16, 16)
        return x
    
class ResBlockUp(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels), 
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(n_filters), 
            nn.ReLU(inplace=False),
        )
        self.residual = UpSampleConv2D(input_channels=n_filters, n_filters=n_filters, kernel_size=3, padding=1)
        self.shortcut = UpSampleConv2D(input_channels=input_channels, n_filters=n_filters, kernel_size=1)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        shortcut = self.shortcut(x)
        x = self.layers(x)
        x = self.residual(x)
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        return torch.add(shortcut, x)

class ResBlockDown(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
        )
        (residual): DownSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): DownSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layer = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=False),
        )
        self.residual = DownSampleConv2D(input_channels=n_filters, n_filters=n_filters, kernel_size=3, padding=1)
        self.shortcut = DownSampleConv2D(input_channels=input_channels, n_filters=n_filters, kernel_size=1)
            
    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        shortcut = self.shortcut(x)
        x = self.layer(x)
        x = self.residual(x)
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        return torch.add(shortcut, x)
    
class ResBlock(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=1),
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        return  self.layer(x)
    
class Generator(jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
        (dense): Linear(in_features=128, out_features=2048, bias=True)
        (layers): Sequential(
            (0): ResBlockUp(
                (layers): Sequential(
                    (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (1): ReLU()
                    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (4): ReLU()
                )
                (residual): UpSampleConv2D(
                    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                )
                (shortcut): UpSampleConv2D(
                    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
                )
            )
            (1): ResBlockUp(
                (layers): Sequential(
                    (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (1): ReLU()
                    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (4): ReLU()
                )
                (residual): UpSampleConv2D(
                    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                )
                (shortcut): UpSampleConv2D(
                    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
                )
            )
            (2): ResBlockUp(
                (layers): Sequential(
                    (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (1): ReLU()
                    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    (4): ReLU()
                )
                (residual): UpSampleConv2D(
                    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                )
                (shortcut): UpSampleConv2D(
                    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
                )
            )
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): Tanh()
        )
    )
    """
    def __init__(self, starting_image_size=4):
        in_feat = 128
        super(Generator, self).__init__()
        self.dense1 = nn.Linear(in_features=in_feat, out_features=2048, bias=True)
        self.layers = nn.Sequential(
            ResBlockUp(in_feat),
            ResBlockUp(in_feat),
            ResBlockUp(in_feat),
            nn.BatchNorm2d(in_feat),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_feat, out_channels=3, kernel_size=1),
            nn.Tanh()
        )
        self.starting_image_size = starting_image_size

    @jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        x = self.dense1(z)
        x = x.view(x.size(0),-1, self.starting_image_size, self.starting_image_size)
        x = self.layers(x)    # [4, 3, 64, 64]
        return x

    @jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        # Make sure to cast the latents to type half (for compatibility with torch.cuda.amp.autocast)        
        noise = torch.randn(n_samples, 128).cuda()
        return (self.forward_given_samples(noise)+1.0)/2.0
    
class Discriminator(jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        n_feat = 128
        self.layers = nn.Sequential(
            ResBlockDown(3, n_filters=n_feat),
            ResBlockDown(n_feat, n_filters=n_feat),
            ResBlock(n_feat, n_filters=n_feat),
            ResBlock(n_feat, n_filters=n_feat),
            nn.ReLU(inplace=False),
        )
        self.dense_out = nn.Linear(in_features=128, out_features=1, bias=True)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to flatten the output of the convolutional layers and sum across the image dimensions before passing to the output layer!
        x = self.layers(x)   # (*, 128, 8, 8)
        x = torch.sum(x.view(x.size(0), x.size(1),-1), dim=-1)
        x = self.dense_out(x.view(x.size(0), -1))
        return x
