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
from dataset.dataset_multi import NYUUWDataset
from tqdm import tqdm
import random
from torchvision import models
import numpy as np
from models.networks_multi import *
import click
import datetime
import pytorch_ssim
from tensorboardX import SummaryWriter
writer = SummaryWriter()
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS


def set_requires_grad(nets, requires_grad=False):
    """
        Make parameters of the given network not trainable
    """

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    return requires_grad


def compute_val_metrics(model, dataloader, no_adv_loss):
    """
        Compute SSIM, PSNR scores for the validation set
    """

    model.eval()

    mse_scores = []
    ssim_scores = []
    psnr_scores = []

    criterion_MSE = nn.MSELoss().cuda()

    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img,  _, structure,detail= data
        uw_img = Variable(uw_img).cuda()
        cl_img = Variable(cl_img, requires_grad=False).cuda()

        structure = Variable(structure).cuda()
        detail = Variable(detail).cuda()

        structure, detail, out, structure_ori,detail3 = model(structure,detail)


        mse_scores.append(criterion_MSE(out, cl_img).item())

        out = (out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

        ssim_scores.append(measure.compare_ssim()(out, cl_img, multichannel=True))
        psnr_scores.append(measure.compare_psnr()(cl_img, out))

        model.train()

    return sum(ssim_scores) / len(dataloader), sum(psnr_scores) / len(dataloader), sum(mse_scores) / len(
        dataloader)



def write_to_log(log_file_path, status):
    """
        Write to the log file
    """

    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')

@click.command()
@click.argument('name')
@click.option('--data_path', default=None, help='Path of training input data')
@click.option('--label_path', default=None, help='Path of training label data')
@click.option('--batch_size', default=4, help='Batch size')
@click.option('--save_interval', default=2, help='Save models after this many epochs')
@click.option('--start_epoch', default=1, help='Start training from this epoch')
@click.option('--end_epoch', default=200, help='Train till this epoch')
@click.option('--num_channels', default=3, help='Number of input image channels')
@click.option('--train_size', default=15000, help='Size of the training dataset')
@click.option('--test_size', default=3000, help='Size of the testing dataset')
@click.option('--val_size', default=3000, help='Size of the validation dataset')
@click.option('--continue_train', is_flag=True, help='Continue training from start_epoch')
@click.option("--snet_lr", default=2e-4, help="structure net Learning Rate. ")
@click.option("--dnet_lr", default=2e-5, help="detail net Learning Rate. ")
def main(name, data_path, label_path,  batch_size, save_interval, start_epoch, end_epoch,
         num_channels,
         train_size, test_size, val_size,continue_train, 
         snet_lr,dnet_lr):


    # Define datasets and dataloaders
    train_dataset = NYUUWDataset(data_path,
                                 label_path,
                                 size=train_size,
                                 train_start=0,
                                 mode='train')

    val_dataset = NYUUWDataset(data_path,
                               label_path,
                               size=val_size,
                               val_start=42000,
                               mode='val')

    test_dataset = NYUUWDataset(data_path,
                                label_path,
                                size=test_size,
                                test_start=46000,
                                mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    # Define models, criterion and optimizers
    model = Dual_cnn().cuda()
    waveGT = WaveGT().cuda()

    criterion_MSE = nn.MSELoss().cuda()
    criterion_MS_SSIM_L1 = MS_SSIM_L1_LOSS().cuda()
    ssim_loss = pytorch_ssim.SSIM().cuda()

    optimizer = torch.optim.RMSprop([{"params": model.snet_structure_fE.parameters(), "lr": snet_lr},
                                     {"params": model.snet_structure_fD.parameters(), "lr": snet_lr},
                                     {"params": model.dnet_detail.parameters(), "lr": dnet_lr},
                                     ])

    waveGT.train()
    model.train()

    if not os.path.exists('./checkpoints/{}'.format(name)):
        os.mkdir('./checkpoints/{}'.format(name))

    log_file_path = './checkpoints/{}/log_file.txt'.format(name)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    status = '\nTRAINING SESSION STARTED ON {}\n'.format(now)
    write_to_log(log_file_path, status)
  

    # Train only the encoder-decoder upto a certain threshold
    for epoch in range(start_epoch, end_epoch):
       
        struct_loss_running = 0.0
        detail_loss_running = 0.0
        recon_loss_running = 0.0
        total_loss_running = 0.0
        n_total_step = len(train_dataloader)

        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img,  _, structure,detail= data
            uw_img = Variable(uw_img).cuda()
            cl_img = Variable(cl_img, requires_grad=False).cuda()

            structure = Variable(structure).cuda()
            detail = Variable(detail).cuda()
            cl_img_structure, cl_img_detail9, cl_img_detail3 = waveGT(cl_img)

            uw_img_structure, _, uw_img_detail3 = waveGT(uw_img)
            optimizer.zero_grad()

            structure, detail, output, structure_ori, detail3 = model(structure,detail)

            struct_loss = 0.5 * criterion_MS_SSIM_L1(structure, cl_img_structure)
            detail_loss = criterion_MSE(detail3, cl_img_detail3)
            recon_loss = 0.2 * (-ssim_loss(output, cl_img))

            loss = recon_loss + struct_loss + detail_loss

            struct_loss_running += struct_loss.item()
            detail_loss_running += detail_loss.item()
            recon_loss_running += recon_loss.item()
            total_loss_running = total_loss_running + loss.item()

            loss.backward()
            progress = "\tEpoch: {}\tIter: {}\tstruct_loss: {}\tdetail_loss: {}\trecon_loss: {}".format(epoch, idx, struct_loss.item(),detail_loss.item(),recon_loss.item())
            optimizer.step()


            if idx % 250 == 0:
                save_image(uw_img.cpu().data, './results/multi/uw_img.png')
                save_image(structure.cpu().data, './results/multi/out_structure.png')
                save_image(detail3.cpu().data, './results/multi/out_detail.png')
                save_image(output.cpu().data, './results/multi/out.png')
                save_image(cl_img_detail3.cpu().data, './results/multi/cl_detail.png')
                save_image(cl_img.cpu().data, './results/multi/cl_img.png')
                save_image(structure_ori.cpu().data, './results/multi/out_structure_ori.png')
                save_image(uw_img_structure.cpu().data, './results/multi/uw_structure.png')
                save_image(uw_img_detail3.cpu().data, './results/multi/uw_detail.png')

                print(progress)
          #      write_to_log(log_file_path, progress)

                writer.add_scalar('struct_loss', struct_loss_running / 250, epoch * n_total_step + idx)
                writer.add_scalar('detail_loss', detail_loss_running / 250, epoch * n_total_step + idx)
                writer.add_scalar('recon_loss', recon_loss_running / 250, epoch * n_total_step + idx)
                writer.add_scalar('total_loss', total_loss_running / 250, epoch * n_total_step + idx)
                struct_loss_running = 0.0
                detail_loss_running = 0.0
                recon_loss_running = 0.0
                total_loss_running = 0.0


                img_grid_fake = torchvision.utils.make_grid(output, normalize=True)
                img_grid_real = torchvision.utils.make_grid(cl_img, normalize=True)
                writer.add_image("Output", img_grid_fake, global_step=idx)
                writer.add_image("GroundTruth", img_grid_real, global_step=idx)


        if epoch % save_interval == 0:
            torch.save(model.state_dict(), './checkpoints/{}/model_{}.pth'.format(name, epoch))

        status = 'End of epoch. Models saved.'
        print(status)
        write_to_log(log_file_path, status)

   #     val_ssim, val_psnr, val_mse= compute_val_metrics(model, val_dataloader, no_adv_loss)
   #     writer.add_scalar('SSIM', val_ssim, epoch)

        start_epoch += 1


if __name__ == "__main__":
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    main()
