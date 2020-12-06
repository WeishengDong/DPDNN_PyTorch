import torch
import numpy as np
from model import DPDNN
import torch.nn as nn
from config import opt
from skimage import io, color
from skimage import measure

test_data = 'Set5'
data_num = 5

with torch.no_grad():
    net = DPDNN()
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(opt.load_model_path))

    psnr_ac = 0
    ssim_ac = 0
    for j in range(data_num):
        # RGB array:[3, H, W]
        label = io.imread(
            './data/test_data/bicubic/X%d/' % opt.scaling_factor + test_data + '/img_%.3d'%(j + 1) + '_SRF_%d_HR'%opt.scaling_factor + '.png')
        test = io.imread(
            './data/test_data/bicubic/X%d/' % opt.scaling_factor + test_data + '/img_%.3d'%(j + 1) + '_SRF_%d_LR'%opt.scaling_factor + '.png')

        if len(label.shape) == 3:
            label_ycbcr = color.rgb2ycbcr(label)
            test_ycbcr = color.rgb2ycbcr(test)
        else:
            label_rgb = color.gray2rgb(label)
            label_ycbcr = color.rgb2ycbcr(label_rgb)
            test_rgb = color.gray2rgb(test)
            test_ycbcr = color.rgb2ycbcr(test_rgb)

        label_y = label_ycbcr[:, :, 0] / 255
        test_y = test_ycbcr[:, :, 0] / 255

        label_cb = label_ycbcr[:, :, 1]
        label_cr = label_ycbcr[:, :, 2]

        label = torch.FloatTensor(label_y).unsqueeze(0).unsqueeze(0).cuda()
        test = torch.FloatTensor(test_y).unsqueeze(0).unsqueeze(0).cuda()

        output = net(test)
        output = torch.clamp(output, 0.0, 1.0)
        loss = (output * 255 - label * 255).pow(2).sum() / (output.shape[2] * output.shape[3])
        psnr = 10 * np.log10(255 * 255 / loss.item())

        output = output.squeeze(0).squeeze(0).cpu()
        label = label.squeeze(0).squeeze(0).cpu()

        output_array = np.array(output * 255).astype(np.float32)
        label_array = np.array(label * 255).astype(np.float32)

        ssim = measure.compare_ssim(output_array, label_array, data_range=255)

        psnr_ac += psnr
        ssim_ac += ssim

        print(j+1, psnr)
    print('PSNR:', psnr_ac/data_num, 'SSIM:', ssim_ac/data_num)





