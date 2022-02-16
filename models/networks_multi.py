import pywt
import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from skimage import measure
import os
from PIL import Image
from tqdm import tqdm
import random
from torchvision import models
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from models.unet_parts import *
import cv2


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""

    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 8)
        self.input = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        return out


class Dual_cnn(nn.Module):

    def __init__(self):
        super(Dual_cnn, self).__init__()

        self.snet_structure_fE = UNetEncoder()
        self.snet_structure_fD = UNetDecoder()
        self.dnet_detail = DNet()
        self.pool = WavePool(3)
        self.LL, self.LH, self.HL, self.HH = get_wav(3, pool=False)

    def forward(self, structure,detail):

        fE_out, enc_out = self.snet_structure_fE(structure)
        structure = self.snet_structure_fD(fE_out, enc_out)
        detail9 = self.dnet_detail(detail)  # (4,9,128,128)
        output = self.LL(structure) + self.LH(detail9[:, 0:3, :, :]) + self.HL(detail9[:, 3:6, :, :]) + self.HH(
            detail9[:, 6:9, :, :])  # (4,3,256,256)
        structure_ori = self.LL(structure)  # (4,3,256,256)
        detail3 = detail9[:, 0:3, :, :] + detail9[:, 3:6, :, :] + detail9[:, 6:9, :, :]  # (4,3,128,128)

        return structure, detail9, output, structure_ori, detail3


class WaveGT(nn.Module):
    def __init__(self):
        super(WaveGT, self).__init__()
        self.pool = WavePool(3)

    def forward(self, x):

        LL, LH, HL, HH = self.pool(x)
        structure = LL
        detail9 = torch.cat([LH, HL, HH], dim=1)  # (4,9,128,128)
        detail3 = LH + HL + HH  # (4,3,128,128)
        return structure, detail9, detail3


class UNetEncoder(nn.Module):
    def __init__(self, n_channels=9):
        super(UNetEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, (x1, x2, x3, x4)


class UNetDecoder(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetDecoder, self).__init__()
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_channels)
        self.sigmoid = nn.Sigmoid()
        self.c = torch.nn.Conv2d(3, 3, 1, padding=0, bias=False)

    def forward(self, x, enc_outs):
        x = self.sigmoid(x)
        x = self.up1(x, enc_outs[3])
        x = self.up2(x, enc_outs[2])
        x = self.up3(x, enc_outs[1])
        x = self.up4(x, enc_outs[0])
        x1 = self.outc(x)

        return x1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 4,64,256,256
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2),  # 4,64,128,128
            nn.Conv2d(32, 64, 3, 1, 1),  # 4,128,128,128
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2)  # 4,128,64,64
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 64 * 64, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )


    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

