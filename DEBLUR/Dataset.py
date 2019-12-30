# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
from data.add_gaussian_noise import add_noise
from config import opt
import cv2


class Train_Data(data.Dataset):
    def __init__(self, data_root):

        self.transform1 = T.ToTensor()
        self.transform2 = T.ToPILImage()
        self.data_root = data_root

    def __getitem__(self, index):
        img_index = random.randint(1, 800)
        img = Image.open(self.data_root + "/" + str(img_index)+'.png')
        img_H = img.size[0]
        img_W = img.size[1]
        H_start = random.randint(0, img_H - opt.crop_size)
        W_start = random.randint(0, img_W - opt.crop_size)
        crop_box = (W_start, H_start, W_start + opt.crop_size, H_start + opt.crop_size)
        img_crop = img.crop(crop_box)    # img_crop:0~255

        label = self.transform1(img_crop)    # label:0~1
        data = np.array(img_crop)   # data:0~255
        # blur = cv2.GaussianBlur(data, (25, 25), 1.6)    # blur:0~255,uint8, ToTensor -> 0~1
        blur = cv2.filter2D(data.astype(np.float32)/255, -1, opt.motion_kernel)     # blur:0~1,float32, ToTensor -> 0~1
        blur = self.transform1(blur)    # blur:0~1
        blur_noise = add_noise(blur, opt.noise_level)   # blur_noise:0~1

        return blur_noise, label

    def __len__(self):
        return opt.num_data


if __name__ == '__main__':
    train_data = Train_Data(data_root=opt.data_root)

    train_loader = DataLoader(train_data, 1)

    transform1 = T.ToPILImage()

    for i, (data, label) in enumerate(train_data):
        print(i)
        if i == 0:
            print(data.size())
            test = transform1(data)
            test.show()
            print(label.size())
            label = transform1(label)
            label.show()

            break





