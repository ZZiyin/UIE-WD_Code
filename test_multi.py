import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from models.networks_multi import *
import os
from PIL import Image
from dataset.dataset_multi import *
from tqdm import tqdm
import random
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from collections import defaultdict
import click

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

def var_to_img(img):
    return (img * 255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

def test(model,  dataloader, model_name, which_epoch):

    with torch.no_grad():
        for idx, data in tqdm(enumerate(dataloader)):
            name, structure,detail= data

            structure = Variable(structure).cuda()
            detail = Variable(detail).cuda()


            structure, detail, output, structure_ori,detail3 = model(structure,detail)
            enc_outs = None

        #    save_image(torch.stack([uw_img.squeeze().cpu().data, output.squeeze().cpu().data,output2.squeeze().cpu().data]), './results/{}/{}/{}_{}.jpg'.format(model_name, which_epoch, name[0], 'out'))
        #    save_image(
        #        torch.stack([uw_img.squeeze().cpu().data,output.squeeze().cpu().data]),
        #        './results/{}/{}/{}_{}.jpg'.format(model_name, which_epoch, name[0], 'out'))


            save_image(
                output.squeeze().cpu().data,
                './results/{}/{}/{}_{}.jpg'.format(model_name, which_epoch, name[0], 'out'))
            """
            save_image(torch.stack([uw_img.squeeze().cpu().data, fI_out.squeeze().cpu().data, cl_img.squeeze().cpu().data]), './results/{}/{}/{}_{}.jpg'.format(model_name, which_epoch, name[0], 'out'))
            mse = criterion_MSE(fI_out, cl_img).item()
            mse_scores.append(mse)

            fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

            ssim = ssim_fn(fI_out, cl_img, multichannel=True)
            psnr = psnr_fn(cl_img, fI_out)

            ssim_scores.append(ssim)
            psnr_scores.append(psnr)

    return ssim_scores, psnr_scores, mse_scores
"""
@click.command()
@click.argument('name')
@click.option('--num_channels', default=3, help='Number of input image channels')
@click.option('--test_dataset', default='nyu', help='Name of the test dataset (nyu)')
@click.option('--data_path', default=None, help='Path of testing input data')
@click.option('--label_path', default=None, help='Path of testing label data')
@click.option('--which_epoch', default=None, help='Test for this epoch')
@click.option('--test_size', default=890, help='The size of test dataset')
@click.option('--model_load_path', default=None, help='Load path for pretrained fN')
def main(name, num_channels, test_dataset, data_path, label_path, which_epoch, test_size, model_load_path):

    if not os.path.exists('./results'):
        os.mkdir('./results')

    if not os.path.exists('./results/{}'.format(name)):
        os.mkdir('./results/{}'.format(name))

    if not os.path.exists('./results/{}/{}'.format(name, which_epoch)):
        os.mkdir('./results/{}/{}'.format(name, which_epoch))


    model = Dual_cnn().cuda()

    model.load_state_dict(torch.load(model_load_path))

    model.eval()

    if test_dataset=='nyu':
        test_dataset = NYUUWDataset(data_path,
            label_path,
            size=3000,
            test_start=33000,
            mode='test')
    else:
        # Add more datasets
        test_dataset = UIEBDataset(data_path,
            label_path,
            size=test_size,
            test_start=0,
            mode='test')

    batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test(model, dataloader, name, which_epoch)
    """
    ssim_scores, psnr_scores, mse_scores = tes(fE, fI, dataloader, name, which_epoch)

    print ("Average SSIM: {}".format(sum(ssim_scores)/len(ssim_scores)))
    print ("Average PSNR: {}".format(sum(psnr_scores)/len(psnr_scores)))
    print ("Average MSE: {}".format(sum(mse_scores)/len(mse_scores)))
"""
if __name__== "__main__":
    main()
