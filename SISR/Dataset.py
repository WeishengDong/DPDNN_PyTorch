

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from config import opt
from skimage import io, color


class Train_Data(data.Dataset):
    def __init__(self):
        self.data_root = opt.data_root
        self.label_root = opt.label_root

    def __getitem__(self, index):
        img_H = io.imread(self.label_root + str(index+1) + '.png')
        img_L = io.imread(self.data_root + str(index+1) + '.png')

        img_H_ycbcr = color.rgb2ycbcr(img_H)
        img_L_ycbcr = color.rgb2ycbcr(img_L)
        img_H_y = img_H_ycbcr[:, :, 0] / 255
        img_L_y = img_L_ycbcr[:, :, 0] / 255

        label = torch.FloatTensor(img_H_y).unsqueeze(0)
        LR_image = torch.FloatTensor(img_L_y).unsqueeze(0)

        return LR_image, label

    def __len__(self):
        return opt.num_data


if __name__ == '__main__':
    train_data = Train_Data()

    train_loader = DataLoader(train_data, 1)

    for i, (data, label) in enumerate(train_data):
        print(i)
        if i == 4:
            data_img = T.ToPILImage()(data)
            data_img.show()
            label_img = T.ToPILImage()(label)
            label_img.show()
            print(data)
            print(label)
            print(data.size())
            print(label.size())
            break












