import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import DPDNN
from Dataset import Train_Data
from config import opt
from torch.utils.data import DataLoader
from visdom import Visdom
import math
from PIL import Image
from torchvision import transforms as T


def train(load_model_path=None):

    train_data = Train_Data(opt.data_root)
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)

    net = DPDNN()
    net = net.cuda()
    net = nn.DataParallel(net)

    # initialize weights by Xavizer
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    if load_model_path:
        net.load_state_dict(torch.load(load_model_path))

    # save model
    torch.save(net.state_dict(), opt.save_model_path)

    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    num_batch = 0
    num_show = 0

    # visdom
    vis = Visdom()

    for epoch in range(opt.max_epoch):
        for i, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            num_batch += 1

            if i % 20 == 0:  # save parameters every 20 batches
                mse_loss, psnr_now = test(epoch, i)
                print('[%d, %5d] loss:%.10f PSNR:%.3f' % (epoch + 1, (i + 1)*opt.batch_size, mse_loss, psnr_now))


                # visdom
                num_show += 1
                x = torch.Tensor([num_show])
                y1 = torch.Tensor([mse_loss])
                y2 = torch.Tensor([psnr_now])
                vis.line(X=x, Y=y1, win='loss', update='append', opts={'title': 'loss'})
                vis.line(X=x, Y=y2, win='PSNR', update='append', opts={'title': 'PSNR'})

                # print('delta1: ', net.delta_1)
                torch.save(net.state_dict(), opt.save_model_path)

        # learning rate decay
        if (epoch+1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])

    print('Finished Training')


def test(epoch, i):
    net1 = DPDNN()
    net1 = net1.cuda()
    net1 = nn.DataParallel(net1)
    net1.load_state_dict(torch.load(opt.load_model_path))

    # for Gaussian blur
    # noise = Image.open('./data/test_data/gaussian_sigma%d/test.png' % opt.noise_level)
    # label = Image.open('./data/test_data/gaussian_sigma%d/testgt.png' % opt.noise_level)

    # for motion blur
    noise = Image.open('./data/test_data/motion%d_sigma%.2f/test.png'%(opt.motion_kernel_size, opt.noise_level))
    label = Image.open('./data/test_data/motion%d_sigma%.2f/testgt.png'%(opt.motion_kernel_size, opt.noise_level))

    img_H = noise.size[0]
    img_W = noise.size[1]

    transform = T.ToTensor()
    transform1 = T.ToPILImage()
    noise = transform(noise)
    noise = noise.resize_(1, 1, img_H, img_W)
    noise = noise.cuda()
    label = np.array(label).astype(np.float32)

    output = net1(noise)     # dim=4
    output = torch.clamp(output, min=0.0, max=1.0)
    output = torch.tensor(output)
    output = output.resize(img_H, img_W).cpu()
    output_img = transform1(output)

    # every 500 batches save test output
    if i%500 == 0:
        save_index = str(int(epoch*(opt.num_data/opt.batch_size/500) + (i+1)/500))
        # for Gaussian blur
        # output_img.save('./data/test_data/gaussian_sigma%d/test_output/'%opt.noise_level+save_index+'.png')

        # for motion blur
        output_img.save('./data/test_data/motion%d_sigma%f/test_output/'%(opt.motion_kernel_size, opt.noise_level)
                        + save_index + '.png')

    output = np.array(output_img)
    mse, psnr = PSNR(output, label)
    return mse, psnr


def PSNR(img1, img2, color=False):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return mse * 255 * 255, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    train()





