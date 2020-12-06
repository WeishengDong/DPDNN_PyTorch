import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import DPDNN
from Dataset import Train_Data
from config import opt
from torch.utils.data import DataLoader
from visdom import Visdom
from PIL import Image
from skimage import io, color
from torchvision import transforms as T
from skimage import measure


def train(load_model_path=None):

    train_data = Train_Data()
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

    num_show = 0
    psnr_best = 0

    # visdom
    vis = Visdom()

    for epoch in range(opt.max_epoch):
        for i, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            # we only need to train Y channel
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:  # save parameters every 20 batches
                mse_loss, psnr_now, ssim = val(net, epoch, i)
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f' % (epoch + 1, (i + 1)*opt.batch_size, mse_loss, psnr_now, ssim))

                # visdom
                num_show += 1
                x = torch.Tensor([num_show])
                y1 = torch.Tensor([mse_loss])
                y2 = torch.Tensor([psnr_now])
                vis.line(X=x, Y=y1, win='loss', update='append', opts={'title': 'loss'})
                vis.line(X=x, Y=y2, win='PSNR', update='append', opts={'title': 'PSNR'})

                if psnr_best < psnr_now:
                    psnr_best = psnr_now
                    torch.save(net.state_dict(), opt.save_model_path)

        # learning rate decay
        if (epoch+1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])

    print('Finished Training')


def val(net1, epoch, i):
    with torch.no_grad():
        psnr_ac = 0
        ssim_ac = 0
        for j in range(5):
            # RGB array:[3, H, W]
            label = io.imread('./data/test_data/bicubic/X%d/'%opt.scaling_factor + 'img_00' + str(j + 1) + '_SRF_%d_HR'%opt.scaling_factor + '.png')
            test = io.imread('./data/test_data/bicubic/X%d/'%opt.scaling_factor + 'img_00' + str(j + 1) + '_SRF_%d_LR'%opt.scaling_factor + '.png')

            label_ycbcr = color.rgb2ycbcr(label)
            test_ycbcr = color.rgb2ycbcr(test)
            label_y = label_ycbcr[:, :, 0] / 255
            test_y = test_ycbcr[:, :, 0] / 255

            label_cb = label_ycbcr[:, :, 1]
            label_cr = label_ycbcr[:, :, 2]

            label = torch.FloatTensor(label_y).unsqueeze(0).unsqueeze(0).cuda()
            test = torch.FloatTensor(test_y).unsqueeze(0).unsqueeze(0).cuda()

            output = net1(test)
            output = torch.clamp(output, 0.0, 1.0)
            loss = (output*255 - label*255).pow(2).sum() / (output.shape[2]*output.shape[3])
            psnr = 10*np.log10(255*255 / loss.item())

            output = output.squeeze(0).squeeze(0).cpu()
            label = label.squeeze(0).squeeze(0).cpu()

            output_array = np.array(output * 255).astype(np.float32)
            label_array = np.array(label * 255).astype(np.float32)
            ssim = measure.compare_ssim(output_array, label_array, data_range=255)

            psnr_ac += psnr
            ssim_ac += ssim

        # every 500 batches save test output
        if i%500 == 0:
            # synthesize SR image
            SR_image = np.zeros([*label_array.shape, 3])
            SR_image[:, :, 0] = output_array
            SR_image[:, :, 1] = label_cb
            SR_image[:, :, 2] = label_cr
            # SR_image = SR_image.astype(np.uint8)
            save_index = str(int(epoch*(opt.num_data/opt.batch_size/500) + (i+1)/500))
            SR_image = color.ycbcr2rgb(SR_image)*255
            SR_image = np.clip(SR_image, a_min=0., a_max=255.)
            SR_image = SR_image.astype(np.uint8)
            io.imsave('./data/test_data/bicubic/X%d/test_output/'%opt.scaling_factor + save_index + '.png', SR_image)

    return loss, psnr_ac/5, ssim_ac/5


if __name__ == '__main__':
    train()





