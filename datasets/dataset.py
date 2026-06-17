import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage import zoom
import random

def is_image_file(filename):
    IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in IMG_EXT)



# ------------------ Dataset class ------------------
class ISIC17Dataset(Dataset):
    def __init__(self, path_Data, config, train=False):
        super().__init__()
        self.train = train
        if self.train:
            img_dir = os.path.join(path_Data, "train/images")
            mask_dir = os.path.join(path_Data, "train/masks")
        else:
            img_dir = os.path.join(path_Data, "val/images")
            mask_dir = os.path.join(path_Data, "val/masks")
        self.transformer = config.train_transformer if train else config.test_transformer

        images_list = sorted([f for f in os.listdir(img_dir) if is_image_file(f)])
        masks_list = sorted([f for f in os.listdir(mask_dir) if is_image_file(f)])

        self.data = []
        for img_name in images_list:
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + "_segmentation.png"

            if mask_name not in masks_list:
                print(f"⚠ Mask not found for {img_name}, skipping...")
                continue

            self.data.append([
                os.path.join(img_dir, img_name),
                os.path.join(mask_dir, mask_name)
            ])

        print(f" Loaded {len(self.data)} image-mask pairs from {img_dir}")

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.expand_dims(np.array(Image.open(mask_path).convert("L")), axis=2) / 255.0


        img, mask = self.transformer((img, mask))

        return img, mask

    def __len__(self):
        return len(self.data)
