
class DefaultConfig(object):

    data_root = './data/DIV2K_gray'
    # label_root = './data/label_128'
    num_data = 450000
    crop_size = 128
    noise_level = 15

    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 1  # how many workers for loading data

    max_epoch = 100
    lr = 0.0005  # initial learning rate
    lr_decay = 0.5

    load_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level
    save_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level


opt = DefaultConfig()


























