import torch
import torch.nn as nn

class DoubleConv(nn.Module): #two blue arrows from arch
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #one blue arrow 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), #next blue arrow 
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module): #two blue arrows + 1 red down arrow
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels) 
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module): #two blue arrows + green arrow
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size =2 , stride = 2) #up convolution (2 x 2) (green arrow)
        self.conv = DoubleConv(in_channels, out_channels)  #blue arrows
    
    def forward(self, x1, x2):
        x1 = self.up(x1) #green arrow
        x = torch.cat([x2, x1], dim = 1) #concat along the channels axis

        return self.conv(x)


        