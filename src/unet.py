import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_parts import DoubleConv, DownSample, UpSample
#double conv block -> 2x2 maxpool downsample (3x) -> bottleneck -> 2x2 upsample  -> double conv block (3x) (concat's across symm layers)


class UNet(nn.Module):
    def __init__(self, in_channels=6, use_line_map = False):
        """
        U-Net for frame interpolation.
        
        Args:
            in_channels: Number of input channels. Default 6 (3 for I0 + 3 for I1).
                        If use_line_map=True, add 1 more channel (total 7).
            use_line_map: Whether to use line map as additional input channel.
        """
        super().__init__()
        self.use_line_map = use_line_map
        
        # Adjust input channels if using line map
        if use_line_map:
            in_channels = in_channels + 1  # Add 1 channel for line map
        
        #ENCODER
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        
        #BOTTLENECK
        self.bottleneck = DoubleConv(512, 1024)

        #DECODER
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        # Output RGB image (3 channels)
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    
    def forward(self, i0, i1, line_map=None):
        """
        Forward pass for frame interpolation.
        
        Args:
            i0: Previous frame (B, 3, H, W) - RGB image
            i1: Next frame (B, 3, H, W) - RGB image
            line_map: Optional line map (B, 1, H, W) - grayscale line map
        
        Returns:
            Predicted middle frame (B, 3, H, W) - RGB image
        """
        # Concatenate I0 and I1 along channel dimension
        x = torch.cat([i0, i1], dim=1)  # (B, 6, H, W)
        
        # Optionally add line map
        if self.use_line_map and line_map is not None:
            x = torch.cat([x, line_map], dim=1)  # (B, 7, H, W)
        
        # Encoder
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(down_1)
        down_3, p3 = self.down_convolution_3(down_2)
        down_4, p4 = self.down_convolution_4(down_3)

        # Bottleneck
        bottleneck = self.bottleneck(down_4)

        # Decoder
        up_1 = self.up_convolution_1(bottleneck, p4)
        up_2 = self.up_convolution_2(up_1, p3)
        up_3 = self.up_convolution_3(up_2, p2)
        up_4 = self.up_convolution_4(up_3, p1)

        # Output RGB image
        out = self.out(up_4)
        return out
