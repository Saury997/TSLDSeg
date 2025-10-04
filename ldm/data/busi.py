import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2
import random


class BUSIBase(Dataset):
    """BUSI Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        self.data_root = data_root
        self.mode = mode
        assert mode in ["training", "val", "test"]
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        print(f"[Dataset]: BUSI with 2 classes, in {self.mode} mode")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        segmentation = Image.fromarray(cv2.cvtColor(
            cv2.imread(example["file_path_"].replace("images", "masks").replace(".png", "_mask.png")),cv2.COLOR_BGR2RGB))
        image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]),cv2.COLOR_BGR2RGB))

        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        segmentation = (np.array(segmentation) > 128).astype(np.float32)
        if self.mode == "test":
            example["segmentation"] = segmentation   
        else:
            example["segmentation"] = ((segmentation * 2) - 1)   # range: binary -1 and 1

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.                            # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self):  # 80% / 10% / 10%
        all_imgs = glob.glob(os.path.join(self.data_root, "*.png"))

        test_path_list = self.data_root + '/' + 'test_images.txt'
        test_imgs = []
        with open(test_path_list, "r") as f:
            for line in f.readlines():
                test_img_path = os.path.join(self.data_root, line.strip().split('\\')[-1])
                test_imgs.append(test_img_path)
                all_imgs.remove(test_img_path)
        train_imgs = all_imgs[:int(len(all_imgs) * 0.8)]
        val_imgs = all_imgs[int(len(all_imgs) * 0.8):int(len(all_imgs) * 0.9)]

        if self.mode == "training":
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


class BUSITrain(BUSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/root/datasets/BUSI/images", mode="training", **kwargs)


class BUSIValidation(BUSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/root/datasets/BUSI/images", mode="val", **kwargs)


class BUSITest(BUSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/root/datasets/BUSI/images", mode="test", **kwargs)

if __name__ == "__main__":
    dataset = BUSITrain(size=256)
    print(dataset[0]['image'].shape)
    print(dataset[0]['segmentation'].shape)
    print(len(dataset))