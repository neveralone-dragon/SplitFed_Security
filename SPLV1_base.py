# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 15:05
# @Author  : Andy_Arthur
# @File    : SPL1_label_flipping.py
# @Software: win10
# ============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T
import time

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
from torchvision.datasets import ImageFolder

import random
import numpy as np
import os

import matplotlib

from Functions.Data_proccess_func import SkinData, dataset_iid, DatasetSplit
from Functions.Fed_algorithm import FedAvg, trimmed_and_thresholded_aggregation, subtract_weights, add_weights, \
    remove_anomalies, krum_aggregation
from Functions.init_func import init_seeds, init_logging, init_models
from Functions.load_data import load_data
from SplitFed_Security.Functions.Models import ResNet18_client_side, ResNet18_server_side, BasicBlock
from Functions.train_test_func import calculate_accuracy, prGreen, prRed
from SplitFed_Security.Functions.class_flipping_df_method import replace_label1_with_label2_on_df, \
    random_select_poisoning_users, poison_data, cifar10_to_dataframe

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
matplotlib.use('Agg')
import copy

SEED = 1234
init_seeds(SEED)

# ===================================================================
program = "SFLV1_ResNet18_base"
print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files
init_logging(program)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# No. of users
num_users = 20
epochs = 100
frac = 1  # participation of clients; if 1 then 100% clients participate in SFLV1
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

# （即权重和偏置）保存到w_glob_server中。
w_glob_server = net_glob_server.state_dict()
w_locals_server = []

# client idx collector
idx_collect = []  # 初始化一个空列表，用于收集选择的客户端的索引。
l_epoch_check = False  # 初始化一个布尔变量，用于指示是否进行了本地训练轮次的检查。
fed_check = False  # 初始化一个布尔变量，用于指示是否完成了联邦学习。
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(num_users)]  # 该列表包含了每个客户端的初始模型。
net_server = copy.deepcopy(net_model_server[0]).to(device)  # 初始化为net_model_server的第一个元素的深拷贝，并将其移到GPU上。


# optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    """

    Args:
        fx_client: 一个函数，用于在客户端更新模型参数，它接受以下参数：net_model_client（客户端模型），optimizer_client（客户端优化器），train_loader（客户端训练数据），l_epoch（客户端训练轮数）。
        y:目标变量的标签值。
        l_epoch_count:训练的总轮数
        l_epoch:当前训练的轮数
        idx:用于选择在全局模型中使用哪些本地模型进行更新的客户端的索引。
        len_batch:训练数据的批次大小。

    Returns:

    """
    # 这些是全局变量，因为它们在函数内被更新，并且在函数之外被调用。
    """
    net_model_server: 全局模型。
    criterion: 损失函数，用于计算模型的误差。
    optimizer_server: 优化器，用于更新全局模型的参数。
    device: 设备（CPU或GPU）用于计算。
    batch_acc_train: 当前批次的准确度。
    batch_loss_train: 当前批次的损失。
    l_epoch_check: 在训练期间用于检查损失和准确度的训练周期数。
    fed_check: 用于检查训练周期是否已完成的标志。
    loss_train_collect: 用于收集所有客户端训练损失的列表。
    acc_train_collect: 用于收集所有客户端训练准确度的列表。
    count1: 计数器，用于跟踪当前已经训练的客户端数量。
    acc_avg_all_user_train: 所有客户端训练准确度的平均值。
    loss_avg_all_user_train: 所有客户端训练损失的平均值。
    idx_collect: 用于跟踪已经训练的客户端的索引列表。
    w_locals_server: 所有客户端本地模型参数的列表。
    w_glob_server: 全局模型参数的列表。
    net_server: 全局模型。
    """
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr

    # net_server是全局模型，返回指定索引的本地模型
    net_server = copy.deepcopy(net_model_server[idx]).to(
        device)
    net_server.train()
    # 是一个PyTorch中的Adam优化器的实现，它接受模型参数和学习率作为参数，用于更新模型参数以最小化损失函数。在这里，我们使用全局模型的参数和一个预定义的学习率 lr 创建了一个Adam优化器对象
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr)

    # 1.train and update
    # 用于清空之前的梯度信息，这样我们可以在每个训练迭代中计算新的梯度并更新模型参数。
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    # ---------forward prop-------------
    fx_server = net_server(fx_client)  # 作为输入传递到全局模型 net_server 中，然后返回模型的预测输出 fx_server

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    loss.backward()
    # 由于我们需要在全局模型更新之前将 fx_client 更新到最新的版本，因此我们使用 clone().detach() 函数来创建一个新的 dfx_client 张量，它具有相同的值但不会被计算图所记录。
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)

    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # 计算当前batch的准确率
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)  # 计算当前batch的损失

        batch_acc_train = []  # 将当前batch准确率清零
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))

        # copy the last trained model in the batch
        # 的状态字典复制到一个新的字典中，以便我们可以将其发送到参与者，从而启动下一轮的联邦学习。注意，w_server 中包含的参数是最新一轮训练的参数，因此每个参与者将从这些参数开始训练它们的本地模型。
        w_server = net_server.state_dict()

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:
            # l_epoch_count 是本地epoch的计数器，l_epoch 是本地epoch的总数。当计数器 l_epoch_count 等于总数 l_epoch 减 1 时，说明本地epoch已经完成。
            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
            # We store the state of the net_glob_server()
            # w_server 是全局模型中最新的训练参数，w_locals_server 是用于存储每个参与者的最后一轮训练参数的列表。因此，当本地epoch完成时，将 w_server 添加到 w_locals_server 中，以便之后将其发送到联邦平均服务器。
            w_locals_server.append(copy.deepcopy(w_server))

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            acc_avg_train_all = acc_avg_train  # 记录最后一个batch的准确率和损失，作为本地epoch的结果
            loss_avg_train_all = loss_avg_train  #

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)  # 将本地epoch的损失添加到损失列表中
            acc_train_collect_user.append(acc_avg_train_all)  # # 将本地epoch的准确率添加到准确率列表中

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)

        # This is for federation process--------------------
        if len(idx_collect) == num_users * frac:
            # 如果客户端编号列表的长度等于客户端总数，说明所有客户端的训练结果都已经到达服务器了。
            # 这里不对，是选择的客户端总数
            fed_check = True  # to evaluate_server function  - to check fed check has hitted

            # 异常检测
            # 使用异常检测移除异常客户端
            # anomaly_threshold = 2  # 自定义阈值
            # w_locals_server = remove_anomalies(w_locals_server, w_glob_server.state_dict(), anomaly_threshold)

            # ================== 使用不同的聚合算法 =================
            w_glob_server = FedAvg(w_locals_server)  # 使用联邦平均算法更新全局模型，将所有客户端的本地模型参数传入该函数中。
            # w_glob_server, selected_clients_indices = krum_aggregation(w_locals_server, 3)

            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)  # 将更新后的全局模型参数加载到服务器端的模型中。
            net_model_server = [net_glob_server for i in
                                range(num_users)]  # 创建一个长度为客户端数量的列表，每个元素都是更新后的全局模型。这个列表用于向每个客户端分发全局模型参数。

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)  # 计算所有客户端训练结果的平均准确率和损失
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            # 更新性能指标列表
            acc_train_collect_user = []
            loss_train_collect_user = []

            w_locals_server = []  # # 清空本地模型参数列表
            idx_collect = []  # 清空客户端编号列表

    # send gradients to the client
    return dfx_client


def evaluate_server(fx_client, y, idx, len_batch, ell, selected_clients):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()
    return_local_results = False

    with torch.no_grad():
        # with torch.no_grad()是一个上下文管理器，它可以暂时关闭所有的requires_grad标志，从而不计算梯度1。这样可以节省内存，提高推理速度，也可以避免不必要的梯度累积2。通常在验证或部署模型时使用这个方法3。
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_server = net(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

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
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")

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
        # DatasetSplit(dataset_train, idxs)表示使用DatasetSplit类将原始的数据集dataset_train按照索引idxs进行划分，以获得当前客户端可用于训练的数据集。
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
                # 内层循环是数据加载器self.ldr_train中每个批次的训练。在每个批次中，将图像和标签加载到设备上，然后将优化器的梯度清零。
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                fx = net(images)
                # 生成一个可求导的副本client_fx
                client_fx = fx.clone().detach().requires_grad_(True)

                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)

                # --------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()

            # prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return net.state_dict()

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
