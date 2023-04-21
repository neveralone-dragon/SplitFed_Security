import os
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from PIL import ImageEnhance

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def add_malicious_samples(self, num_malicious_samples, dataset_type='CIFAR10'):
        for i in range(num_malicious_samples):
            # Randomly select an image to modify
            dataset_idx = random.choice(self.idxs)

            # Access the image data depending on the dataset structure
            img = self.dataset.data[dataset_idx]
            img = Image.fromarray(img)

            # Apply a random modification to the image (e.g., flip, rotate, or change color)
            modified_img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip the image horizontally
            modified_img = modified_img.rotate(random.randint(0, 360))  # Rotate the image randomly

            brightness = ImageEnhance.Brightness(modified_img)
            modified_img = brightness.enhance(random.uniform(0.5, 1.5))

            contrast = ImageEnhance.Contrast(modified_img)
            modified_img = contrast.enhance(random.uniform(0.5, 1.5))

            saturation = ImageEnhance.Color(modified_img)
            modified_img = saturation.enhance(random.uniform(0.5, 1.5))

            # Add additional distortions
            modified_img = modified_img.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))  # Apply Gaussian blur

            # Add noise to the image
            noise = np.random.normal(0, 25, modified_img.size)
            noise = noise.reshape(modified_img.size[::-1]).T.astype(np.uint8)
            modified_img = Image.fromarray(np.array(modified_img) + noise)

            # Clip the pixel values to be within the valid range (0-255)
            modified_img = np.clip(modified_img, 0, 255)

            # Convert the modified PIL Image back to a numpy array
            modified_img = np.array(modified_img)

            self.dataset.data[dataset_idx] = modified_img

    def add_malicious_samples_on_HAM10000(self, num_malicious_samples, dataset_type='HAM10000'):
        # 遗憾的是没有改变
        for i in range(num_malicious_samples):
            # Randomly select an image to modify
            dataset_idx = random.choice(self.idxs)

            # Access the image data depending on the dataset structure
            img = self.dataset.data[dataset_idx]
            img = Image.fromarray(img)

            # Convert the image to a numpy array
            img_array = np.array(img)

            # Define the black region size
            black_region_height = int(0.8 * img_array.shape[0])
            black_region_width = int(0.8 * img_array.shape[1])

            # Define the starting point for the black region
            start_height = int(0.1 * img_array.shape[0])
            start_width = int(0.1 * img_array.shape[1])

            # Set the region to black
            img_array[start_height:start_height + black_region_height,
            start_width:start_width + black_region_width] = 0


            # Update the image in the dataset
            self.dataset.data[dataset_idx] = img_array


# class SkinData(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, index):
#         X = Image.open(self.df['path'][index]).resize((64, 64))
#         y = torch.tensor(int(self.df['target'][index]))
#
#         if self.transform:
#             X = self.transform(X)
#
#         return X, y

class SkinData(Dataset):
    def __init__(self, df, data, targets, transform=None):
        self.df = df
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, target


def dataset_iid(dataset, num_users):
    """
    该函数接受一个数据集dataset和一个整数num_users作为输入。
    它的作用是将数据集分割成num_users份，以便每个客户端都有一份相同分布的数据集。
    :param dataset:
    :param num_users:
    :return:函数返回一个字典dict_users，其中包含num_users个键，每个键对应一个客户端，值为该客户端所分配的数据集索引的集合。
    dict_users:{idx:int : []:list}
    """
    # 该函数首先计算每个客户端应该拥有的数据量num_items
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # 接着，函数使用np.random.choice函数从all_idxs中选择num_items个索引，将这些索引添加到字典dict_users的第i个键中，表示第i个客户端的数据集。在选择后，从all_idxs中移除已经分配给第i个客户端的索引。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users