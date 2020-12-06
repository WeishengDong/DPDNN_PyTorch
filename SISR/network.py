# coding=UTF-8

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in ):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=3 // 2)

        self.act = torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        f_e = self.act(self.conv3(out2))
        down = self.act(self.conv4(f_e))
        return f_e, down

class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in = 64 ):
        super(Encoding_Block_End, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2)
        self.act =  torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


    def forward(self, input):

        f_e = self.act(self.conv(input))
        return f_e

class Decoding_Block(torch.nn.Module):
    def __init__(self, c_in ):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.batch = 1
        # self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=3 // 2)
        self.act =  torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input,label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])
        # kernel = autograd.Variable(torch.randn(64,64,3,3))
        #
        # Deconv = F.conv2d_transpose(input, kernel,
        #                                 output_shape=[batch_size, label_h1, label_h2, out_features],
        #                                 strides=[1, 2, 2, 1], padding='SAME')
        Deconv = self.up(input)

        return Deconv

    def forward(self, input, map):

        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        up = self.act(up)
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        decoding = self.act(self.conv4(out3))

        return decoding

class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=3, padding=3 // 2)
        self.act =  torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
    def forward(self, input):

        conv = self.conv0(input)
        return conv


class DPDNN(torch.nn.Module):
    """
    network of 'Burst Denoising with Kernel Prediction Networks'
    """
    def __init__(self , channel0, factor):
        super(DPDNN, self).__init__()
        # self.K = K
        # self.N = N
        self.channel0 = channel0
        self.up_factor = factor

        self.Encoding_block1 = Encoding_Block(channel0)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)


        self.Encoding_block_end = Encoding_Block_End(64)


        self.Decoding_block1 = Decoding_Block(64)
        self.Decoding_block2 = Decoding_Block(64)
        self.Decoding_block3 = Decoding_Block(64)
        self.Decoding_block4 = Decoding_Block(64)

        self.feature_decoding_end = Feature_Decoding_End(channel0)
        self.UP = torch.nn.Upsample(scale_factor=factor,mode='bilinear')
        self.down = torch.nn.Upsample(scale_factor=1/factor,mode='bilinear')


        self.acti =  torch.nn.PReLU()
        self.delta = torch.nn.Parameter(torch.tensor(0.1))
        self.eta = torch.nn.Parameter(torch.tensor(0.9))
        self.reset_parameters()

    def recon(self, features, recon, noise, SR):
        recon_h1 = int(recon.shape[2])
        recon_h2 = int(recon.shape[3])
        # delta = torch.tensor(0.1).cuda()
        # eta = torch.tensor(0.9).cuda()


        # down = bicubic_interp_2d(recon, [int(recon_h1 / SR), int(recon_h2 / SR)])
        # err1 = bicubic_interp_2d(down - noise, [recon_h1, recon_h2])

        down = torch.nn.functional.interpolate(recon, scale_factor=1/self.up_factor, mode='bicubic', align_corners=False)
        err1 = torch.nn.functional.interpolate(down - noise, scale_factor=self.up_factor, mode='bicubic', align_corners=False)

        err2 = recon - features
        out = recon - self.delta * (err1 + self.eta * err2)
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)




    def forward(self,  input): # [batch_size ,3 ,7 ,270 ,480] ;
        label_h1 = int(input.shape[2])*self.up_factor
        label_h2 = int(input.shape[3])*self.up_factor

        # x = bicubic_interp_2d(input, [label_h1, label_h2])
        x = torch.nn.functional.interpolate(input, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        # x1 = self.UP(input)

        # x = x.permute(0,3,1,2)
        y = input

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block4(decode1, encode0)

        decoding_end = self.feature_decoding_end(decode0)
        #x = decoding_end
        conv_out = x + decoding_end
        # x  =  conv_out
        x = self.recon(conv_out, x, y, SR=2)

        for i in range(5):

            encode0, down0 = self.Encoding_block1(x)
            encode1, down1 = self.Encoding_block2(down0)
            encode2, down2 = self.Encoding_block3(down1)
            encode3, down3 = self.Encoding_block4(down2)

            media_end = self.Encoding_block_end(down3)

            decode3 = self.Decoding_block1(media_end,encode3)
            decode2 = self.Decoding_block2(decode3, encode2)
            decode1 = self.Decoding_block3(decode2, encode1)
            decode0 = self.Decoding_block4(decode1, encode0)

            decoding_end = self.feature_decoding_end(decode0)
            conv_out = x + decoding_end

            x = self.recon(conv_out, x, y, SR=2)
        #
        return x


if __name__ == '__main__':

    input1 = torch.rand(1, 1, 114, 172)

    net = DPDNN(1, 2)

    out = net(input1)
    # print(net)
    print(out.size())

    # print(net.delta_1)

