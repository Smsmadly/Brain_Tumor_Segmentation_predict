''' model.py '''

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # First
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            # Second
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ConvTranspose(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvTranspose, self).__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, 
                kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), output_padding=0, bias=True
            ),
            nn.LeakyReLU()            
        )

    def forward(self, x):
        return self.conv_transpose(x)

class FCN(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(FCN, self).__init__()
        # Init
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Downsamplers
        self.dp1 = nn.MaxPool2d((2, 2))
        self.dp2 = nn.MaxPool2d((2, 2))
        self.dp3 = nn.MaxPool2d((2, 2))
        self.dp4 = nn.MaxPool2d((2, 2))
        # Double Convolution Blocks
        ## Down
        self.dc1 = DoubleConv(in_channels, 64)
        self.dc2 = DoubleConv(64, 128)
        self.dc3 = DoubleConv(128, 256)
        self.dc4 = DoubleConv(256, 512)
        self.dc5 = DoubleConv(512, 1024)
        ## Up
        self.uc4 = DoubleConv(512 * 2, 512)
        self.uc3 = DoubleConv(256 * 2, 256)
        self.uc2 = DoubleConv(128 * 2, 128)
        self.uc1 = DoubleConv(64 * 2, 64)
        # Upsamplers
        self.up4 = ConvTranspose(1024, 512)
        self.up3 = ConvTranspose(512, 256)
        self.up2 = ConvTranspose(256, 128)
        self.up1 = ConvTranspose(128, 64)
        # Final Convolutional layer
        self.fc = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # Encoder
        
        dc1 = self.dc1(x)
        dp1 = self.dp1(dc1)
        
        dc2 = self.dc2(dp1)
        dp2 = self.dp2(dc2)
        
        dc3 = self.dc3(dp2)
        dp3 = self.dp3(dc3)
        
        dc4 = self.dc4(dp3)
        dp4 = self.dp4(dc4)
        
        dc5 = self.dc5(dp4) # Lowest block
        
        # Decoder
        
        up4 = self.up4(dc5)
        cat = torch.cat([up4, dc4], dim=1)
        uc4 = self.uc4(cat)
        
        up3 = self.up3(uc4)
        cat = torch.cat([up3, dc3], dim=1)
        uc3 = self.uc3(cat)

        up2 = self.up2(uc3)
        cat = torch.cat([up2, dc2], dim=1)
        uc2 = self.uc2(cat)

        up1 = self.up1(uc2)
        cat = torch.cat([up1, dc1], dim=1)
        uc1 = self.uc1(cat)
        
        # Final Convolution Layer
        
        out = self.fc(uc1)
        out = self.sigmoid(out)
        
        return out