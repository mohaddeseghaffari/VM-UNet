import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import os
import random
import logging
import logging.handlers
import matplotlib

from matplotlib import pyplot as plt


# ---------------------------
#  بدون SimpleITK و MedPy
# ---------------------------
def compute_dice(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()

    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def compute_hd95(pred, gt):
    """
    برای 2D ISIC چون medpy نداریم یک نسخه ساده‌شده HD95 بر اساس فاصلهٔ لبه‌ها
    """

    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0

    from scipy.spatial.distance import cdist
    dists = cdist(pred_pts, gt_pts)
    return np.percentile(dists.min(axis=1), 95)



# ---------------------------
# Seed
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Logger
# ---------------------------
def get_logger(name, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, f"{name}.info.log")
    info_handler = logging.handlers.TimedRotatingFileHandler(
        info_name, when='D', encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    return logger


def log_config_info(config, logger):
    logger.info('#----------Config info----------#')
    for k, v in config.__dict__.items():
        if k.startswith('_'): continue
        logger.info(f'{k}: {v}')


# ---------------------------
# Optimizer + Scheduler
# ---------------------------
def get_optimizer(config, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )


def get_scheduler(config, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.T_max,
        eta_min=config.eta_min
    )


# ---------------------------
# Save Result Images
# ---------------------------
from PIL import Image
import numpy as np
import os
import torch

def save_imgs(img, msk, msk_pred, i, save_path, datasets=None, threshold=0.5, test_data_name=None):
    from PIL import Image
    import numpy as np
    import os
    import torch

    os.makedirs(save_path, exist_ok=True)

    # img
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 255)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = img.astype(np.uint8)

    # msk
    if isinstance(msk, torch.Tensor):
        msk = msk.squeeze().cpu().numpy()
    # تبدیل به 2 بعدی
    if msk.ndim > 2:
        msk = msk.squeeze()  # حذف ابعاد اضافی
    msk = (msk * 255).astype(np.uint8)

    # msk_pred
    if isinstance(msk_pred, torch.Tensor):
        msk_pred = msk_pred.squeeze().cpu().numpy()
    if msk_pred.ndim > 2:
        msk_pred = msk_pred.squeeze()
    msk_pred = (msk_pred > threshold).astype(np.uint8) * 255

    name = f"{test_data_name}_{i}.png" if test_data_name else f"{i}.png"

    Image.fromarray(img).save(os.path.join(save_path, f"img_{name}"))
    Image.fromarray(msk).save(os.path.join(save_path, f"gt_{name}"))
    Image.fromarray(msk_pred).save(os.path.join(save_path, f"pred_{name}"))

# ---------------------------
# Losses
# ---------------------------
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        pred = pred.view(-1)
        target = target.view(-1)
        inter = (pred * target).sum()
        return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        return self.wb * self.bce(pred, target) + self.wd * self.dice(pred, target)


# ---------------------------
# Transforms
# ---------------------------
class myToTensor:
    def __call__(self, data):
        img, msk = data
        return torch.tensor(img).permute(2, 0, 1), torch.tensor(msk).permute(2, 0, 1)


class myResize:
    def __init__(self, h=256, w=256):
        self.h = h
        self.w = w
    def __call__(self, data):
        img, msk = data
        return TF.resize(img, (self.h, self.w)), TF.resize(msk, (self.h, self.w))


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        img, msk = data
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(msk)
        return img, msk


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        img, msk = data
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(msk)
        return img, msk


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.p = p
        self.degree = degree

    def __call__(self, data):
        img, msk = data
        if random.random() < self.p:
            angle = random.uniform(*self.degree)
            return TF.rotate(img, angle), TF.rotate(msk, angle)
        return img, msk


class myNormalize:
    def __init__(self, dataset, train=True):
        if dataset == 'isic17':
            self.mean = 159.922 if train else 148.429
            self.std  = 28.871  if train else 25.748
        else:
            self.mean, self.std = 150, 30

    def __call__(self, data):
        img, msk = data
        img_norm = (img - self.mean) / self.std
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
        img_norm = img_norm * 255.0
        return img_norm, msk


from thop import profile		 ## 导入thop模块
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')


