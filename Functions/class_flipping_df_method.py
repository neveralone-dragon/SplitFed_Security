#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 22:31
# @Author  : Andy_Arthur
# @File    : class_flipping_df_method.py
# @Software: win10
import random

import numpy as np
import pandas as pd


def replace_label1_with_label2_on_df(df,label1,label2,poisoned_dict_users):
    """
    标签反转
    :param df:dataframe
    :param label1: 等待翻转的标签
    :param label2: 需要翻转的标签
    :param poisoned_dict_users:包含索引列表的字典,中毒的用户
    :return:
    """
    for idx_list in poisoned_dict_users.values():
        for idx in idx_list:
            if df.loc[idx,'target'] == label1:
                df.loc[idx,'target'] = label2
    # df.loc[df['target'] == label1, 'target'] = label2
    return df

def random_select_poisoning_users(dict_users, n):
    """
    随机选择n个key-value对
    :param dict_users: 字典，key为索引，value为包含索引值的列表
    :param n: 选择的key-value对的数量
    :return: 随机选择的key-value对组成的字典
    """
    selected = {}
    keys = random.sample(list(dict_users.keys()), n)
    for k in keys:
        selected[k] = dict_users[k]
    return selected

# def poison_data(dataset_train, dict_users, poisoned_users_num, original_label, target_label):
#     poisoned_dict_users = random_select_poisoning_users(dict_users, poisoned_users_num)
#     replace_label1_with_label2_on_df(dataset_train.df, original_label, target_label, poisoned_dict_users)
#     return poisoned_dict_users

def print_poisoning_results(poisoned_dict_users, dataset_train):
    for poisoned_user_key in poisoned_dict_users:
        print("被投毒的用户:", poisoned_user_key)

    print("标签反转后的target统计:")
    print(dataset_train.df['target'].value_counts())

def cifar10_to_dataframe(dataset):
    data = [dataset[i] for i in range(len(dataset))]
    images, labels = zip(*data)
    df = pd.DataFrame({"image": images, "target": labels})
    return df



def poison_data(dataset_train, dataset_choice, dict_users, poisoned_users_num, label_mappings):
    if dataset_choice == 'CIFAR10':
        df_train = cifar10_to_dataframe(dataset_train)
    elif dataset_choice == 'HAM10000':
        df_train = dataset_train.df
    else:
        raise ValueError("Invalid dataset choice.")

    poisoned_dict_users = random_select_poisoning_users(dict_users, poisoned_users_num)

    for original_label, target_label in label_mappings:
        replace_label1_with_label2_on_df(df_train, original_label, target_label, poisoned_dict_users)

    # 修改 dataset_train 的标签
    if dataset_choice == 'CIFAR10':
        for idx, row in df_train.iterrows():
            dataset_train.targets[idx] = row['target']

    return poisoned_dict_users

def poison_data_random(dataset_train, dataset_choice, dict_users, poisoned_users_num):
    if dataset_choice == 'CIFAR10':
        df_train = cifar10_to_dataframe(dataset_train)
    elif dataset_choice == 'HAM10000':
        df_train = dataset_train.df
    else:
        raise ValueError("Invalid dataset choice.")

    poisoned_dict_users = random_select_poisoning_users(dict_users, poisoned_users_num)

    for idx_list in poisoned_dict_users.values():
        for idx in idx_list:
            labels = list(set(df_train['target'].unique()) - {df_train.loc[idx, 'target']})
            df_train.loc[idx, 'target'] = random.choice(labels)

    # 修改 dataset_train 的标签
    if dataset_choice == 'CIFAR10':
        for idx, row in df_train.iterrows():
            dataset_train.targets[idx] = row['target']

    return poisoned_dict_users

def poison_data_model(dataset_train, dataset_choice, dict_users, poisoned_users_num, attack_pattern, target_label):
    if dataset_choice == 'CIFAR10':
        df_train = cifar10_to_dataframe(dataset_train)
    elif dataset_choice == 'HAM10000':
        df_train = dataset_train.df
    else:
        raise ValueError("Invalid dataset choice.")

    poisoned_dict_users = random_select_poisoning_users(dict_users, poisoned_users_num)

    for idx_list in poisoned_dict_users.values():
        for idx in idx_list:
            if dataset_choice == 'CIFAR10':
                dataset_train.data[idx] = attack_pattern
            elif dataset_choice == 'HAM10000':
                # Assuming the image data is stored in a 'images' attribute in the HAM10000 dataset
                dataset_train.images[idx] = blend_images(dataset_train.images[idx], attack_pattern)
            df_train.loc[idx, 'target'] = target_label

    # 修改 dataset_train 的标签
    if dataset_choice == 'CIFAR10':
        for idx, row in df_train.iterrows():
            dataset_train.targets[idx] = row['target']
    elif dataset_choice == 'HAM10000':
        for idx, row in df_train.iterrows():
            dataset_train.labels[idx] = row['target']

    return poisoned_dict_users

def blend_images(image1, image2, alpha=0.5):
    return np.clip(image1 * alpha + image2 * (1 - alpha), 0, 255).astype(np.uint8)

def generate_label_mappings(dataset_choice):
    if dataset_choice == 'CIFAR10':
        num_classes = 10
    elif dataset_choice == 'HAM10000':
        num_classes = 7
    else:
        raise ValueError("Invalid dataset choice.")

    label_mappings = []
    for i in range(num_classes):
        target_label = random.choice([j for j in range(num_classes) if j != i])
        label_mappings.append((i, target_label))

    return label_mappings