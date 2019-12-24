
import torch
from PIL import Image 
import numpy as np
from torchvision import transforms as T
from utils import add_noise
from utils import PSNR
import math
from model import DPDNN
import torch.nn as nn
from config import opt

# Here is the path of your test image, 'i' means the ith image, you only need to provide the ground truth image
# Then we add Gaussian noise to the gt image
i = 1
label_img = './Set12/%.2d.png'%i


transform1 = T.ToTensor()
transform2 = T.ToPILImage()

with torch.no_grad():
    net = DPDNN()
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(opt.load_model_path))

    img = Image.open(label_img)
    # img.show()
    label = np.array(img).astype(np.float32)   # label:0~255
    img_H = img.size[0]
    img_W = img.size[1]
    img = transform1(img)

    img_noise = add_noise(img, opt.noise_level).resize_(1, 1, img_H, img_W)

    output = net(img_noise)
    output = output.cpu()
    output = output.resize_(img_H, img_W)
    output = torch.clamp(output, min=0, max=1)
    output = transform2(output)

    # output.show()
    # To save the output(denoised) image, you must create a new folder. Here is my path.
    output.save('./output/sigma%d/%d.png'%(opt.noise_level, i))

    img_noise = transform2(img_noise.resize_(img_H, img_W))
    # img_noise.show()
    img_noise.save('./output/sigma%d/%d_noise.png'%(opt.noise_level, i))
    output = np.array(output)   # output:0~255

    # Because of the randomness of Gaussian noise, the output results are different each time.
    print(i, 'MSE loss:%f, PSNR:%f'%(PSNR(output, label)))










