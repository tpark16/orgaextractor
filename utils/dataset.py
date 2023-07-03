import os
from cv2 import GaussianBlur
import numpy as np

import torch
import torch.nn as nn
import cv2
import random

import torchvision.transforms.functional as TF
from torchvision import transforms

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train_transform=True):
        self.data_dir = data_dir
        self.train_transform = train_transform

        lst_data = os.listdir(self.data_dir)

        self.lst_label = [f for f in lst_data if f.startswith('label')]
        self.lst_input = [f for f in lst_data if f.startswith('input')]
        self.lst_label.sort()
        self.lst_input.sort()

    def __len__(self):
        return len(self.lst_label) if self.train_transform else len(self.lst_input)
    
    def transform(self, image, mask):
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        if random.random() > 0.5:
            image = TF.affine(image, angle=15, translate=(0.2, 0.2), scale=1.5, shear=15)
            mask = TF.affine(mask, angle=15, translate=(0.2, 0.2), scale=1.5, shear=15)


        # Gaussian Blur
        if random.random() > 0.5:
            image = TF.gaussian_blur(image, (3,3))
            mask = TF.gaussian_blur(mask, (3,3))

        image = TF.normalize(image, 0.5, 0.5)

        return image, mask

    # for test
    def test_transform(self, image):
        # Transform to tensor
        image = TF.to_tensor(image)

        image = TF.normalize(image, 0.5, 0.5)

        return image


    def __getitem__(self, index):

        p = os.path.join(self.data_dir, self.lst_label[index])

        if p.endswith('npy'):
            input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
            label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        else:
            input = cv2.imread(os.path.join(self.data_dir, self.lst_input[index]), 0)
            label = cv2.imread(os.path.join(self.data_dir, self.lst_label[index]), 0)

        if self.train_transform:
            label = label/255.0
            input = input/255.0

            if label.ndim == 2:
                label = label[:, :, np.newaxis]
            if input.ndim == 2:
                input = input[:, :, np.newaxis]
        else:
            input = input/255.0
            if input.ndim == 2:
                input = input[:, :, np.newaxis]

        if self.train_transform:
            input, label = self.transform(input, label)
            return input, label
        else: 
            input = self.test_transform(input)
            return input