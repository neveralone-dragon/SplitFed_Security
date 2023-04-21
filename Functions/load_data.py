#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/26 21:59
# @Author  : Andy_Arthur
# @File    : load_data.py
# @Software: win10
import os

import numpy as np
import pandas as pd

from glob import glob

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from Functions.Data_proccess_func import SkinData, dataset_iid, DatasetSplit
import torchvision.datasets as datasets



def load_data(dataset_choice, num_users):
    """

    :param dataset_choice: 选择的数据集
    :param num_users: 用户的数量
    :return:
    """
    if dataset_choice == 'HAM10000':
        df = pd.read_csv('data/HAM10000_metadata.csv')
        print(df.head())

        lesion_type = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }

        # merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
        imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join("data", '*', '*.jpg'))}

        # print("path---------------------------------------", imageid_path.get)
        # 将图像id映射为图像文件的路径，并将其存储在数据集中的path列中。
        df['path'] = df['image_id'].map(imageid_path.get)
        # 将诊断编码映射为对应的分类名称，并将其存储在数据集中的cell_type列中。
        df['cell_type'] = df['dx'].map(lesion_type.get)
        # 将分类名称转换为数字编码，并将其存储在数据集中的target列中。这里使用了.
        # 可以将字符串类型的分类变量转换为数字编码，其中不同的分类名称对应不同的数字编码。
        df['target'] = pd.Categorical(df['cell_type']).codes
        print(df['cell_type'].value_counts())
        print(df['target'].value_counts())

        # =============================================================================
        # Train-test split
        train, test = train_test_split(df, test_size=0.2)

        train = train.reset_index()

        test = test.reset_index()

        # Load image data and targets
        image_data = []
        for path in train['path']:
            image = Image.open(path).resize((64, 64))
            image_data.append(np.array(image))
        train_data = np.array(image_data)
        train_targets = train['target'].astype(np.int64).to_numpy()  # Change dtype to int64

        image_data = []
        for path in test['path']:
            image = Image.open(path).resize((64, 64))
            image_data.append(np.array(image))
        test_data = np.array(image_data)
        test_targets = test['target'].astype(np.int64).to_numpy()  # Change dtype to int64



        # =============================================================================
        #                         Data preprocessing
        # =============================================================================
        # Data preprocessing: Transformation
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.Pad(3),
                                               transforms.RandomRotation(10),
                                               transforms.CenterCrop(64),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=mean, std=std)
                                               ])

        test_transforms = transforms.Compose([
            transforms.Pad(3),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # With augmentation
        dataset_train = SkinData(train, train_data, train_targets, transform=train_transforms)
        dataset_test = SkinData(train, test_data, test_targets, transform=test_transforms)

        # ----------------------------------------------------------------
        dict_users = dataset_iid(dataset_train, num_users)
        dict_users_test = dataset_iid(dataset_test, num_users)
    elif dataset_choice == 'CIFAR10':
        # =============================================================================
        #                         Data loading
        # =============================================================================
        # Load CIFAR-10 dataset
        trainset = datasets.CIFAR10(root='./data', train=True, download=True)
        testset = datasets.CIFAR10(root='./data', train=False, download=True)

        train_df = pd.DataFrame(trainset.targets, columns=['target'])
        test_df = pd.DataFrame(testset.targets, columns=['target'])

        # Set the class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        train_df['cell_type'] = train_df['target'].apply(lambda x: class_names[x])
        test_df['cell_type'] = test_df['target'].apply(lambda x: class_names[x])

        print(train_df['cell_type'].value_counts())
        print(train_df['target'].value_counts())

        # =============================================================================
        # Train-test split
        train = train_df.reset_index()
        test = test_df.reset_index()

        # =============================================================================
        #                         Data preprocessing
        # =============================================================================
        # Data preprocessing: Transformation
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.Pad(3),
                                               transforms.RandomRotation(10),
                                               transforms.CenterCrop(32),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=mean, std=std)
                                               ])

        test_transforms = transforms.Compose([
            transforms.Pad(3),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # With augmentation
        dataset_train = datasets.CIFAR10(root='./data', train=True, transform=train_transforms, download=True)
        dataset_test = datasets.CIFAR10(root='./data', train=False, transform=test_transforms, download=True)

        # ----------------------------------------------------------------
        dict_users = dataset_iid(dataset_train, num_users)
        dict_users_test = dataset_iid(dataset_test, num_users)
    else:
        raise ValueError("Invalid dataset_choice: Choose either 'HAM10000' or 'CIFAR10'")

    return dataset_train,dataset_test,dict_users,dict_users_test
