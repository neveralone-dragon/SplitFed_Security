#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 10:16
# @Author  : Andy_Arthur
# @File    : SPLV2_base_Resnet.py
# @Software: win10
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T
import time

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation
# ============================================================================
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame

import torchvision.datasets as datasets
import random
import numpy as np
import os
from torchvision.datasets import ImageFolder

import matplotlib

from Functions.Data_proccess_func import SkinData, dataset_iid, DatasetSplit
from Functions.Fed_algorithm import FedAvg, krum_aggregation
from Functions.init_func import init_seeds, init_logging, init_models
from Functions.load_data import load_data
from Functions.train_test_func import calculate_accuracy, prGreen, prRed
from SplitFed_Security.Functions.Models import ResNet18_client_side, ResNet18_server_side, BasicBlock
from SplitFed_Security.Functions.class_flipping_df_method import replace_label1_with_label2_on_df, \
    random_select_poisoning_users, poison_data, print_poisoning_results, cifar10_to_dataframe

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

SEED = 1234
init_seeds(SEED)

# ===================================================================
program = "SFLV2_ResNet18_base"
print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files
init_logging(program)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# No. of users
num_users = 20
epochs = 100
frac = 1  # participation of clients; if 1 then 100% clients participate in SFLV2
lr = 0.0001

# CIFAR10\HAM10000
dataset_choice = 'CIFAR10'
net_glob_client, net_glob_server = init_models(device, dataset_choice)

# ===================================================================================
# For Server Side Loss and Accuracy
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

# ====================================================================================================
#                                  Server Side Programs
# ====================================================================================================
# Federated averaging: FedAvg
# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

# client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False


# Server-side function associated with Training
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    """

    :param fx_client:客户端计算得到的特征值
    :param y:对应的标签
    :param l_epoch_count:本地训练周期计数
    :param l_epoch:本地训练周期总数
    :param idx:
    :param len_batch:
    :return:
    """
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user, lr

    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    # ---------forward prop-------------
    # 将fx_client输入到服务器端模型net_glob_server，计算得到预测值fx_server
    fx_server = net_glob_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    loss.backward()
    # 将fx_client的梯度保存为dfx_client。
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # server-side model net_glob_server is global so it is updated automatically in each pass to this function

    # count1: to track the completion of the local batch associated with one client
    count1 += 1  # 增加计数器count1，用于追踪与一个客户端关联的本地批次的完成情况。
    if count1 == len_batch:
        # 如果count1等于len_batch，则计算批次的平均损失和准确度，并将它们分别存储在acc_avg_train和loss_avg_train中。然后清空
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:

            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)

            # print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)
                # print(idx_collect)

        # This is to check if all users are served for one round --------------------
        if len(idx_collect) == num_users * frac:
            fed_check = True  # to evaluate_server function  - to check fed check has hitted
            # all users served for one round ------------------------- output print and update is done in evaluate_server()
            # for nicer display

            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client


# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell, selected_clients):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net_glob_server.eval()
    return_local_results = False

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_server = net_glob_server(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        # 计数器 count2 用于跟踪处理的客户端数量。如果已经处理了一个批次的客户端数据，那么计算批次的平均损失和平均准确度，然后重置批次损失和批次准确度列表以及计数器。
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                             loss_avg_test))

            # if a local epoch is completed
            if l_epoch_check:
                l_epoch_check = False
                return_local_results = True

                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if all users are served for one round ----------
            if fed_check:
                fed_check = False

                # 计算Krum选定客户端的平均准确率和损失
                if selected_clients is None or len(selected_clients) == 0:
                    acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                    loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)
                else:
                    print("选择的客户端index:", selected_clients)
                    acc_test_collect_user = [acc_test_collect_user[i] for i in selected_clients]
                    loss_test_collect_user = [loss_test_collect_user[i] for i in selected_clients]

                    acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                    loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    if return_local_results:
        return acc_avg_test_all, loss_avg_test_all
    else:
        return None, None


class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                 idxs_test=None, is_attacker=None, batch_size=128):
        """
        :param idxs: idxs是一个表示该客户端用于训练的数据集的索引列表。在联邦学习中，原始数据集通常由多个客户端持有，每个客户端只能访问自己所持有的部分数据集。因此，为了让每个客户端只使用自己所持有的数据进行训练，需要将原始数据集划分成多个部分，每个部分由一个客户端持有，并通过idxs将该客户端用于训练的数据集的索引列表传递给Client类的构造函数。
        :param idxs_test:
        """
        # net_client_model:一个与客户端实例相关的神经网络模型。
        self.batch_size = batch_size
        self.is_attacker = is_attacker
        self.idx = idx  # 一个整数，表示客户端的索引
        self.device = device  # 一个字符串，表示执行客户端计算的设备。
        self.lr = lr
        self.local_ep = 1
        # self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=self.batch_size,
                                    shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=self.batch_size, shuffle=True)

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

        for iter in range(self.local_ep):
            if self.is_attacker:
                dataset_split = self.ldr_train.dataset
                num_malicious_samples = len(dataset_split.idxs) // 2
                dataset_split.add_malicious_samples(num_malicious_samples)

                # Refresh the DataLoader after adding malicious samples
                self.ldr_train = DataLoader(dataset_split, batch_size=self.batch_size, shuffle=True)

            # 外层循环是客户端的本地训练轮数self.local_ep
            len_batch = len(self.ldr_train)  # 计算该客户端的训练集数据分成的批次数。
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                # 循环处理每个批次的训练数据，batch_idx表示该批次的索引，images和labels表示该批次的图像和标签。
                # 内层循环是数据加载器self.ldr_train中每个批次的训练。在每个批次中，将图像和标签加载到设备上，然后将优化器的梯度清零。
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                fx = net(images)
                # 生成一个可求导的副本client_fx
                # 对预测值进行深度拷贝，得到一个可求导的副本client_fx，并设置该张量需要计算梯度，以供后续使用。
                # 因此，在训练过程中，首先对本地模型参数进行一次前向传递，得到预测值，然后使用clone()方法生成一个与预测值张量具有相同数值但独立于原始张量的新张量client_fx，并通过detach()方法使其与计算图断开连接，最后设置requires_grad_属性为True，使得该张量具有求导的能力，从而在之后的反向传递中可以计算损失函数对该张量的梯度，以供服务器使用。
                client_fx = fx.clone().detach().requires_grad_(True)

                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)

                # --------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()

            # prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return net.state_dict()

    def add_malicious_samples(self, images, labels):
        # Modify a portion of the images in the batch to create malicious samples
        num_malicious_samples = len(images) // 2  # Modify 20% of the images

        for i in range(num_malicious_samples):
            # Randomly select an image to modify
            img_idx = random.randint(0, len(images) - 1)
            img = images[img_idx]

            # Apply a random modification to the image (e.g., flip, rotate, or change color)
            # In this example, we flip the image horizontally
            modified_img = torch.flip(img, dims=[2])

            # Update the image in the batch
            images[img_idx] = modified_img

            # Change the label of the modified image to a random class
            # Note: You may want to change this to a specific target class, depending on your attack goal
            random_label = random.randint(0, 9)  # Assuming 10 classes in the dataset
            labels[img_idx] = random_label

        return images, labels

    def evaluate(self, net=None, ell=None, selected_clients=None):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)

                # Sending activations to server
                acc_avg_test_all, loss_avg_test_all = evaluate_server(fx, labels, self.idx, len_batch, ell,
                                                                      selected_clients)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            if loss_avg_test_all is not None and acc_avg_test_all is not None:
                self.loss_avg_test_all = loss_avg_test_all
                self.acc_avg_test_all = acc_avg_test_all

        return


# =============================================================================
#                         Data loading
# =============================================================================
dataset_train, dataset_test, dict_users, dict_users_test = load_data(dataset_choice, num_users)

# =============================================================================
#                         Poisoning
# =============================================================================
# 增加投毒用户数量至 50%
# poisoned_frac = 0.3
# poisoned_users_num = int(poisoned_frac * num_users)
# label_mappings = [(4, 2), (1, 7), (3, 1), (2, 6), (6, 5)]
# poisoned_dict_users = poison_data(dataset_train, dataset_choice, dict_users, poisoned_users_num, label_mappings)
#
# for poisoned_user_key in poisoned_dict_users:
#     print("被投毒的用户:", poisoned_user_key)
#
# # 如果使用的是 HAM10000 数据集，您可以直接使用 dataset_train.df 查看标签分布
# if dataset_choice == 'HAM10000':
#     print("标签反转后的target统计:")
#     print(dataset_train.df['target'].value_counts())
# elif dataset_choice == 'CIFAR10':
#     cifar10_df = cifar10_to_dataframe(dataset_train)
#     print("标签反转后的target统计:")
#     print(cifar10_df['target'].value_counts())

# ------------ Training And Testing -----------------
net_glob_client.train()
# copy weights
w_glob_client = net_glob_client.state_dict()

# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

total_time = 0.0  # 初始化总时间为0
best_clients_indices = None
krum_acc_test_collect = []
krum_loss_test_collect = []

# 梯度匹配攻击
# Choose a fraction of clients to be attackers
attackers_frac = 0.4  # Modify this value according to your requirements
num_attackers = int(attackers_frac * num_users)
attackers_indices = np.random.choice(num_users, num_attackers, replace=False)
print("匹配攻击选取的客户端",str(attackers_indices))

for iter in range(epochs):
    start_time = time.time()  # 记录开始时间
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)  # ，replace=False表示不允许重复选择。
    w_locals_client = []  # 用于存储每个客户端训练后的本地模型参数。
    loss_avg_test_all_dict = {}
    acc_avg_test_all_dict = {}
    idx_local_client_dict = {}

    for idx in idxs_users:
        is_attacker = idx in attackers_indices
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx], is_attacker=is_attacker)
        # Training ------------------
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        idx_local_client_dict[len(w_locals_client)] = idx
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter, selected_clients=best_clients_indices)

        # Update the dictionaries with the client's self.loss_avg_test_all and self.acc_avg_test_all
        loss_avg_test_all_dict[idx] = local.loss_avg_test_all
        acc_avg_test_all_dict[idx] = local.acc_avg_test_all

    print("idxs_users", idxs_users)
    # Ater serving all clients for its local epochs------------
    # Federation process at Client-Side------------------------
    print("------------------------------------------------------------")
    print("------ Fed Server: Federation process at Client-Side -------")
    print("------------------------------------------------------------")
    # w_locals_client是所有客户端训练后的本地模型参数列表，FedAvg函数是加权平均函数，返回全局模型参数w_glob_client。
    w_glob_client = FedAvg(w_locals_client)
    # 调用 Krum 算法
    # num_to_select = int(num_users * (1 - poisoned_frac - 0.1))  # 选择的客户端数量
    # w_glob_client, best_clients_indices = krum_aggregation(w_locals_client, num_to_select)
    # print([idxs_users[i] for i in best_clients_indices])
    # Update client-side global model
    net_glob_client.load_state_dict(w_glob_client)

    # print("fedserver选择的客户端index:", best_clients_indices)
    # best_clients_idxs = [idx_local_client_dict[i] for i in best_clients_indices]
    # krum_acc_test_collect_user = [acc_avg_test_all_dict[i] for i in best_clients_idxs]
    # for acc in krum_acc_test_collect_user:
    #     print("acc:",acc)
    # krum_loss_test_collect_user = [loss_avg_test_all_dict[i] for i in best_clients_idxs]
    #
    # krum_acc_avg_all_user = sum(krum_acc_test_collect_user) / len(krum_acc_test_collect_user)
    # krum_loss_avg_all_user = sum(krum_loss_test_collect_user) / len(krum_loss_test_collect_user)
    # krum_acc_test_collect.append(krum_acc_avg_all_user)
    # krum_loss_test_collect.append(krum_loss_avg_all_user)

    # print("====================== Fed Server==========================")
    # print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_all_user_train,
    #                                                                           loss_avg_all_user_train))
    # print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, krum_acc_avg_all_user,
    #                                                                          krum_loss_avg_all_user))
    # print("==========================================================")

    end_time = time.time()  # 记录结束时间
    epoch_time = end_time - start_time  # 计算epoch所耗费的时间
    total_time += epoch_time  # 将时间差加到总时间中
    # 将时间差值转换为小时、分钟和秒数
    hours, rem = divmod(epoch_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Epoch {iter} finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Epoch {iter} finished. Total time: {total_time:.2f} seconds")

# ===================================================================================

print("Training and Evaluation completed!")

# ===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect) + 1)]
df = DataFrame({'round': round_process, 'acc_train': acc_train_collect, 'acc_test': acc_test_collect,
                'loss_train': loss_train_collect, 'loss_test': loss_test_collect})
file_name = program + "_" + dataset_choice + ".xlsx"
df.to_excel(file_name, sheet_name="v1_test", index=False)

# =============================================================================
#                         Program Completed
# =============================================================================
