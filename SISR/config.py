
import scipy.io as io
import random


class DefaultConfig(object):
    scaling_factor = 2
    LR_size = 32
    HR_size = LR_size*scaling_factor

    data_root = './data/train_data/bicubic/X%d/LR_%d/'%(scaling_factor, LR_size)
    label_root = './data/train_data/bicubic/X%d/HR_%d/'%(scaling_factor, HR_size)

    num_data = 200000

    batch_size = 4  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 0  # how many workers for loading data

    max_epoch = 100
    lr = 0.0005  # initial learning rate
    lr_decay = 0.5

    load_model_path = './checkpoints/DPDNN_SISR_bicubic%d.pth' % (scaling_factor)
    save_model_path = './checkpoints/DPDNN_SISR_bicubic%d.pth' % (scaling_factor)


opt = DefaultConfig()

























