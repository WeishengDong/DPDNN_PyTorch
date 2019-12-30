# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:07:10 2019

@author: 31723
"""

import torch
from PIL import Image 
import numpy as np
from torchvision import transforms as T
from utils import add_noise, PSNR
from model import DPDNN
import torch.nn as nn
from config import opt

import cv2

i = 1
label_img = './Set10/%.2d.png'%i


transform1 = T.ToTensor()
transform2 = T.ToPILImage()

with torch.no_grad():
    net = DPDNN()
    net = nn.DataParallel(net)

    # For different blur kernels, you should select 'load_model_path_Gaussian' or 'load_model_path_motion'
    # and Modify the corresponding settings in config.py
    # --------------------------------------------------------
    net.load_state_dict(torch.load(opt.load_model_path_Gaussian))
    # --------------------------------------------------------

    img = Image.open(label_img)
    # img.show()
    label = np.array(img).astype(np.float32)   # label:0~255
    img_H = img.size[0]
    img_W = img.size[1]

    img_data = np.array(img)

    # -------------------------------------------------------------------------------
    # for Gaussian blur
    img_blur = cv2.GaussianBlur(img_data, (25, 25), 1.6)    # img_blur : uint8 , 0~255
    img_blur = transform1(img_blur)    # img_blur : tensor, 0~1

    # for motion blur
    # img_blur = cv2.filter2D(img_data.astype(np.float32)/255, -1, opt.motion_kernel)
    # img_blur = transform1(img_blur)
    # -------------------------------------------------------------------------------

    # add Gaussian noise
    img_blur = add_noise(img_blur, opt.noise_level).resize_(1, 1, img_H, img_W)

    output = net(img_blur)
    output = output.cpu()
    output = output.resize_(img_H, img_W)
    output = torch.clamp(output, min=0, max=1)
    output = transform2(output)

    img_blur = transform2(img_blur.resize_(img_H, img_W))
    img_blur.show()
    # img_blur.save('./output/motion%d_sigma%.2f/%d_blur.png' % (opt.motion_kernel_size, opt.noise_level, i))

    output.show()
    # save image respectively, use your path, not mine
    # output.save('./output/gaussian_sigma%d/%d.png'%(opt.noise_level, i))
    # output.save('./output/motion%d_sigma%.2f/%d.png' % (opt.motion_kernel_size, opt.noise_level, i))
    output = np.array(output)   # output:0~255

    # Because of the randomness of Gaussian noise, the output results are different each time.
    print(i, 'MSE loss:%f, PSNR:%f' % (PSNR(output, label)))










