
import scipy.io as io


class DefaultConfig(object):

    data_root = './data/DIV2K_gray'

    num_data = 480000
    crop_size = 128
    noise_level = 2
    motion_kernel_size = 19

    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data

    max_epoch = 100
    lr = 0.0005  # initial learning rate
    lr_decay = 0.5

    # for Gaussian blur
    load_model_path_Gaussian = './checkpoints/DPDNN_gaussian_sigma%d.pth'%noise_level
    save_model_path_Gaussian = './checkpoints/DPDNN_gaussian_sigma%d.pth'%noise_level

    # for motion blur
    load_model_path_motion = './checkpoints/DPDNN_motion%d_sigma%.2f.pth' % (motion_kernel_size, noise_level)
    save_model_path_motion = './checkpoints/DPDNN_motion%d_sigma%.2f.pth' % (motion_kernel_size, noise_level)

    if motion_kernel_size == 19:
        kernel_path = './data/LevinEtalCVPR09Data/Levin09blurdata/im05_flit0%.1d.mat'%1

    if motion_kernel_size == 17:
        kernel_path = './data/LevinEtalCVPR09Data/Levin09blurdata/im05_flit0%.1d.mat'%2

    motion_kernel = io.loadmat(kernel_path)['f']


opt = DefaultConfig()

























