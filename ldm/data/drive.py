#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
* Author: Zongjian Yang
* Date: 2024/12/31 下午3:21 
* Project: Stable-Diffusion-Seg 
* File: drive.py
* IDE: PyCharm 
* Function:
"""
import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class DRIVEBase(Dataset):
    """DRIVE Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=512, interpolation="nearest", mode=None, num_classes=2):
        self.data_root = data_root
        self.mode = mode
        assert mode in ["train", "val", "test"]
        self.data_paths = self._parse_data_list()
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.CenterCrop(size=(256, 256))
        ])

        print(f"[Dataset]: DRIVE with 2 classes, in {self.mode} mode")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)

        # 使用 Pillow 替代 cv2 进行加载
        segmentation_path = example["file_path_"].replace("image", "gt").replace(".tif", ".gif")
        segmentation = Image.open(segmentation_path).convert("RGB")

        image_path = example["file_path_"]
        image = Image.open(image_path).convert("RGB")

        # 调整图像大小
        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        # 训练模式下执行额外的变换
        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        # 将分割图像转换为二值化的 numpy 数组
        segmentation = (np.array(segmentation) > 128).astype(np.float32)
        if self.mode == "test":
            example["segmentation"] = segmentation
        else:
            example["segmentation"] = ((segmentation * 2) - 1)  # 二值化范围调整为 -1 和 1

        # 将图像数据归一化到 -1 到 1
        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.
        example["image"] = image

        # 类别 ID，二分类任务中无实际影响
        example["class_id"] = np.array([-1])

        # 断言范围检查
        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.

        return example

    def __len__(self):
        return len(self.data_paths)

    def _parse_data_list(self): # 80% / 10% / 10%
        all_imgs = glob.glob(os.path.join(self.data_root, "*.tif")) + glob.glob(os.path.join(self.data_root, "*.gif"))
        train_imgs, val_imgs, test_imgs = all_imgs[:], all_imgs[30:36], all_imgs[36:]

        if self.mode == "train":
            return train_imgs
        elif self.mode == "val":
            return val_imgs
        elif self.mode == "test":
            return test_imgs
        else:
            raise NotImplementedError(f"Only support dataset split: train, val, test !")

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class DRIVETrain(DRIVEBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/root/datasets/DRIVE/image", mode="train", **kwargs)


class DRIVEValidation(DRIVEBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/root/datasets/DRIVE/image", mode="val", **kwargs)


class DRIVETest(DRIVEBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/root/datasets/DRIVE/image", mode="test", **kwargs)

if __name__ == "__main__":
    dataset = DRIVETrain(size=512)
    print(dataset[0]['image'].shape)