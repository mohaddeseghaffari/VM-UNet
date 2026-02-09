import sys
import os

# اضافه کردن مسیر root پروژه
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from torchvision import transforms
from utils import *
from datetime import datetime


class setting_config:
    """
    Full config (training + testing), but we only use testing part.
    """

    # ---------------------- Model ----------------------
    network = 'vmunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.0,
        'load_ckpt_path': "./pre_trained_weights/vmamba_tiny_e292.pth",      # no pretrained backbone
    }

    # ---------------------- Dataset ----------------------
    datasets = 'isic17'
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic17/'
    else:
        raise Exception('datasets is not correct!')

    criterion = BceDiceLoss(wb=1, wd=1)

    # ---------------------- Sizes ----------------------
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3

    # ---------------------- System ----------------------
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    gpu_id = '0'
    amp = False

    # ---------------------- Training (ignored) ----------------------
    batch_size = 8
    epochs = 25
    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    # ---------------------- Testing ----------------------
    only_test_and_save_figs = False

    # !!! IMPORTANT: FIXED PATHS FOR COLAB !!!
    best_ckpt_path = './best-chpt/best-vmunet-isic17.pth'
    img_save_path  = './img'

    # ---------------------- Transforms ----------------------
    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])

    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    # ---------------------- Optimizer ----------------------
    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD']

    if opt == 'AdamW':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False

    # ---------------------- Scheduler ----------------------
    sch = 'CosineAnnealingLR'
    if sch == 'CosineAnnealingLR':
        T_max = 50
        eta_min = 1e-5
        last_epoch = -1

    # ---------------------- Log directory ----------------------
    work_dir = 'results/' + network + '_' + datasets + '_' + \
        datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
