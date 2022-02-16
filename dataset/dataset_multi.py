import torch
from torch.utils.data.dataset import Dataset
from glob import glob
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
import cv2
from torch import nn
import torch.nn.functional as F
import scipy.misc
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

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

def set_channel(*l, only_y=False):
    def _set_channel(img):
        return bgr2ycbcr(img, only_y=only_y)

    return [_set_channel(_l) for _l in l]


def np2tensor(*args, pixel_range):
    def _np2tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(pixel_range / 255)

        return tensor

    return [_np2tensor(arg) for arg in args]

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class NYUUWDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png', size=30000, mode='train', train_start=0, val_start=30000, test_start=33000):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.size = size
        self.train_start = train_start
        self.test_start = test_start
        self.val_start = val_start

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))  # glob get each image path in the data file, return a list([])

        if self.mode == 'train':
            self.uw_images = self.uw_images[self.train_start:self.train_start+self.size]
        elif self.mode == 'test':
            self.uw_images = self.uw_images[self.test_start:self.test_start+self.size]
        elif self.mode == 'val':
            self.uw_images = self.uw_images[self.val_start:self.val_start+self.size]

        self.cl_images = []  # label path + image number + .png

        for img in self.uw_images:
            self.cl_images.append(os.path.join(self.label_path, os.path.basename(img).split('_')[0]  + '.' + img_format))

        for uw_img, cl_img in zip(self.uw_images, self.cl_images):
            assert os.path.basename(uw_img).split('_')[0] == os.path.basename(cl_img).split('.')[0], ("Files not in sync.")

        self.transform = transforms.Compose([
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):

            uw_img = self.transform(Image.open(self.uw_images[index]))
            cl_img = self.transform(Image.open(self.cl_images[index]))

            wavePool = WavePool(3)
            uw_img = uw_img.unsqueeze(0)
            LL, LH, HL, HH = wavePool(uw_img)
            uw_img = np.squeeze(uw_img, axis=0)

            LL = LL.numpy()
            LL = np.squeeze(LL, axis=0)
            LL = np.transpose(LL,(1,2,0))

            LL_LAB = cv2.cvtColor(LL, cv2.COLOR_RGB2LAB)
            LL_HSV = cv2.cvtColor(LL, cv2.COLOR_RGB2HSV)
            LL_RGB = LL
            LL_LAB = torch.from_numpy(LL_LAB)
            LL_HSV = torch.from_numpy(LL_HSV)
            LL_RGB = torch.from_numpy(LL_RGB)

            structure = torch.cat([LL_RGB, LL_LAB, LL_HSV], dim=2)
            structure = np.transpose(structure, (2,0,1))

            detail = torch.cat([LH, HL, HH], dim=1)
            name = os.path.basename(self.uw_images[index])[:-4]
            detail = np.squeeze(detail, axis=0)

            return uw_img, cl_img, name, structure, detail

    def __len__(self):
        return self.size

class UIEBDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png', size=30000, mode='train', train_start=0, val_start=30000, test_start=33000):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.size = size
        self.train_start = train_start
        self.test_start = test_start
        self.val_start = val_start

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))

        if self.mode == 'train':
            self.uw_images = self.uw_images[self.train_start:self.train_start+self.size]
        elif self.mode == 'test':
            self.uw_images = self.uw_images[self.test_start:self.test_start+self.size]
        elif self.mode == 'val':
            self.uw_images = self.uw_images[self.val_start:self.val_start+self.size]

        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        uw_img = self.transform(Image.open(self.uw_images[index]))

        wavePool = WavePool(3)
        uw_img = uw_img.unsqueeze(0)
        LL, LH, HL, HH = wavePool(uw_img)
        uw_img = np.squeeze(uw_img, axis=0)

        LL = LL.numpy()
        LL = np.squeeze(LL, axis=0)
        LL = np.transpose(LL, (1, 2, 0))

        LL_LAB = cv2.cvtColor(LL, cv2.COLOR_RGB2LAB)
        LL_HSV = cv2.cvtColor(LL, cv2.COLOR_RGB2HSV)
        LL_RGB = LL
        LL_LAB = torch.from_numpy(LL_LAB)
        LL_HSV = torch.from_numpy(LL_HSV)
        LL_RGB = torch.from_numpy(LL_RGB)

        structure = torch.cat([LL_RGB, LL_LAB, LL_HSV], dim=2)
        structure = np.transpose(structure, (2, 0, 1))

        detail = torch.cat([LH, HL, HH], dim=1)
        name = os.path.basename(self.uw_images[index])[:-4]
        detail = np.squeeze(detail, axis=0)

        return os.path.basename(self.uw_images[index]), structure, detail

    def __len__(self):
        return self.size