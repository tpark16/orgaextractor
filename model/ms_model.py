from __future__ import absolute_import, print_function
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

class Residual_block_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_block_3, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=1,
                                bias=True)]
        layers += [nn.InstanceNorm2d(num_features=out_channels)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=1,
                                bias=True)]
        layers += [nn.InstanceNorm2d(num_features=out_channels)]
        layers += [nn.ReLU()]

        self.conv = nn.Sequential(*layers)

        skips = []
        skips += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=3, stride=1, padding=1,
                            bias=True)]
        skips += [nn.InstanceNorm2d(num_features=out_channels)]

        self.skip = nn.Sequential(*skips)

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        return x


class Residual_block_7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_block_7, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=7, stride=1, padding=3,
                                bias=True)]
        layers += [nn.InstanceNorm2d(num_features=out_channels)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=7, stride=1, padding=3,
                                bias=True)]
        layers += [nn.InstanceNorm2d(num_features=out_channels)]
        layers += [nn.ReLU()]

        self.conv = nn.Sequential(*layers)

        skips = []
        skips += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=7, stride=1, padding=3,
                            bias=True)]
        skips += [nn.InstanceNorm2d(num_features=out_channels)]

        self.skip = nn.Sequential(*skips)

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        return x

        
class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_block, self).__init__()
        self.x3 = Residual_block_3(in_channels, out_channels)
        self.x7 = Residual_block_7(in_channels, out_channels)

        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x3 = self.x3(x)
        x7 = self.x7(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x  

class ResUNet_MS(nn.Module):
    def __init__(self):
        super(ResUNet_MS, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Contracting path
        self.enc1_1 = Residual_block(in_channels=1, out_channels=64)
        # self.input_block = input_block(in_channels=1, out_channels=64) 
        # self.input_skip = input_skip(in_channels=1, out_channels=64)

        # self.enc1_2 = Residual_block(in_channels=64, out_channels=64)


        self.enc2_1 = Residual_block(in_channels=64, out_channels=128)
        # self.enc2_2 = Residual_block(in_channels=128, out_channels=128)

        # self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = Residual_block(in_channels=128, out_channels=256)
        # self.enc3_2 = Residual_block(in_channels=256, out_channels=256)

        # self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = Residual_block(in_channels=256, out_channels=512)
        # self.enc4_2 = Residual_block(in_channels=512, out_channels=512)

        # self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = Residual_block(in_channels=512, out_channels=1024)

        # Expansive path
        self.unpool5 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec5_1 = Residual_block(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # self.dec4_2 = Residual_block(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = Residual_block(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # self.dec3_2 = Residual_block(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = Residual_block(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # self.dec2_2 = Residual_block(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = Residual_block(in_channels=128, out_channels=64)

        # self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
        #                                   kernel_size=2, stride=2, padding=0, bias=True)
        # # self.dec1_2 = Residual_block(in_channels=2 * 64, out_channels=64)
        # self.dec1_1 = Residual_block(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        # enc1_1 = self.input_block(x) + self.input_skip(x)
        # pool1 = self.pool(enc1_1)
        # enc1_2 = self.enc1_2(pool1)

        pool2 = self.pool(enc1_1)
        enc2_1 = self.enc2_1(pool2)
        # enc2_2 = self.enc2_2(enc2_1)

        pool3 = self.pool(enc2_1)
        enc3_1 = self.enc3_1(pool3)
        # enc3_2 = self.enc3_2(enc3_1)

        pool4 = self.pool(enc3_1)
        enc4_1 = self.enc4_1(pool4)
        # enc4_2 = self.enc4_2(enc4_1)

        pool5 = self.pool(enc4_1)
        enc5_1 = self.enc5_1(pool5)

        unpool5 = self.unpool5(enc5_1)
        cat5 = torch.cat((unpool5, enc4_1), dim=1)
        dec5_1 = self.dec5_1(cat5)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc3_1), dim=1)
        # dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(cat4)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc2_1), dim=1)
        # dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(cat3)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc1_1), dim=1)
        # dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(cat2)

        # unpool1 = self.unpool1(dec2_1)
        # cat1 = torch.cat((unpool1, enc1_2), dim=1)
        # # dec1_2 = self.dec1_2(cat1)
        # dec1_1 = self.dec1_1(cat1)

        x = self.fc(dec2_1)

        return x

if __name__ == "__main__":

    import torch
    import torchvision
    from torchsummary import summary

    device = torch.device('cpu')
    model = ResUNet_MS().to(device)
    summary(model, (1,256,256), batch_size=1, device='cpu')

    