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
from collections import OrderedDict


import torchvision.datasets as datasets
import random
import numpy as np
import os
from torchvision.datasets import ImageFolder

import matplotlib

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

SEED = 1234

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))
init_seeds(SEED)

from torch import nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResNet18_client_side(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet18_client_side, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pool_size=4):  # Add a new argument pool_size
        super(ResNet18_server_side, self).__init__()
        self.in_planes = 64
        self.pool_size = pool_size  # Add this line to store pool_size
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        # print("Output shape before pooling:", out.shape)
        out = F.avg_pool2d(out, kernel_size=self.pool_size)  # Use self.pool_size instead of 8
        # print("Output shape after pooling:", out.shape)
        out = out.view(out.size(0), -1)
        y_hat = self.linear(out)
        return y_hat


import logging


def init_logging(program_name):
    logging.basicConfig(filename=f'{program_name}.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


# ===================================================================
program = "SFLV1_label_random_"
print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files
init_logging(program)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================

# No. of users
num_users = 10
epochs = 100
frac = 1  # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.0001
roni_threshold = 0.5

# CIFAR10\HAM10000
dataset_choice = 'CIFAR10'

def init_models(device, dataset_choice):
    net_glob_client = ResNet18_client_side(BasicBlock, [2, 2, 2, 2]).to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"We use {torch.cuda.device_count()} GPUs")
        net_glob_client = nn.DataParallel(net_glob_client)

    net_glob_client.to(device)
    logging.info(net_glob_client)

    if dataset_choice == 'HAM10000':
        num_classes = 7
        pool_size = 8
    elif dataset_choice == 'CIFAR10':
        num_classes = 10
        pool_size = 4
    else:
        raise ValueError('Invalid dataset choice.')

    net_glob_server = ResNet18_server_side(BasicBlock, [2,2,2,2], num_classes=num_classes,pool_size=pool_size)
    if torch.cuda.device_count() > 1:
        logging.info(f"We use {torch.cuda.device_count()} GPUs")
        net_glob_server = nn.DataParallel(net_glob_server)

    net_glob_server.to(device)
    logging.info(net_glob_server)

    return net_glob_client, net_glob_server

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
#                                  Server Side Program
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

    # net_server是全局模型，返回制定索引的本地模型
    net_server = copy.deepcopy(net_model_server[idx]).to(
        device)  # copy.deepcopy() 函数用于创建一个当前本地模型的副本，以便我们可以在全局模型的更新过程中使用它，而不会对原始本地模型进行更改。
    # 方法将模型设置为训练模式，这意味着在计算时会使用训练期间的正则化技术，如dropout或batch normalization。
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
            # # 标记已经完成本地epoch
            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
            # We store the state of the net_glob_server()
            # w_server 是全局模型中最新的训练参数，w_locals_server 是用于存储每个参与者的最后一轮训练参数的列表。因此，当本地epoch完成时，将 w_server 添加到 w_locals_server 中，以便之后将其发送到联邦平均服务器。
            w_locals_server.append(copy.deepcopy(w_server))

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)

            # print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train  # 记录最后一个batch的准确率和损失，作为本地epoch的结果
            loss_avg_train_all = loss_avg_train  #

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)  # 将本地epoch的损失添加到损失列表中
            acc_train_collect_user.append(acc_avg_train_all)  # # 将本地epoch的准确率添加到准确率列表中

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)
                # print(idx_collect)
#                 print("已经训练的客户端:" + str(idx_collect))

        # This is for federation process--------------------
        if len(idx_collect) == num_users * frac:
            fed_check = True  # to evaluate_server function  - to check fed check has hitted
            # ================== 使用不同的聚合算法 =================
            print("检测前的local长度",len(w_locals_server))
            # apply RONI to filter out abnormal clients
            w_locals_server = apply_roni(w_locals_server, w_glob_server, dataset_test, net_glob_server, device, roni_threshold)
            print("检测后的local长度",len(w_locals_server))
            w_glob_server = FedAvg(w_locals_server)  # 使用联邦平均算法更新全局模型，将所有客户端的本地模型参数传入该函数中。

            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)  # 将更新后的全局模型参数加载到服务器端的模型中。
            net_model_server = [net_glob_server for i in
                                range(num_users)]  # 创建一个长度为客户端数量的列表，每个元素都是更新后的全局模型。这个列表用于向每个客户端分发全局模型参数。



            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)  # 计算所有客户端训练结果的平均准确率和损失
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            # 更新性能指标列表
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

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

            # if federation is happened----------
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
                 idxs_test=None):
        # net_client_model:一个与客户端实例相关的神经网络模型。
        self.idx = idx  # 一个整数，表示客户端的索引
        self.device = device  # 一个字符串，表示执行客户端计算的设备。
        self.lr = lr
        self.local_ep = 1
        # self.selected_clients = []
        # DatasetSplit(dataset_train, idxs)表示使用DatasetSplit类将原始的数据集dataset_train按照索引idxs进行划分，以获得当前客户端可用于训练的数据集。
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=128,
                                    shuffle=True)  # 一个PyTorch数据集，表示客户端可用于训练的数据
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=128, shuffle=True)

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
        for iter in range(self.local_ep):
            # 外层循环是客户端的本地训练轮数self.local_ep
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # 内层循环是数据加载器self.ldr_train中每个批次的训练。在每个批次中，将图像和标签加载到设备上，然后将优化器的梯度清零。
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                fx = net(images)
                # 生成一个可求导的副本client_fx
                client_fx = fx.clone().detach().requires_grad_(True)

                # Sending activations to server and receiving gradients from server
                # dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)


                # --------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()

            # prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return net.state_dict()

    def evaluate(self, net, ell, selected_clients=None):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)

                # Sending activations to server
                acc_avg_test_all, loss_avg_test_all = evaluate_server(fx, labels, self.idx, len_batch, ell, selected_clients)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            if loss_avg_test_all is not None and acc_avg_test_all is not None:
                self.loss_avg_test_all = loss_avg_test_all
                self.acc_avg_test_all = acc_avg_test_all

        return

from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from PIL import ImageEnhance

from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# ==============================================================
# Custom dataset prepration in Pytorch format
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
                        for x in glob(os.path.join("../data", '*', '*.jpg'))}

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

dataset_train, dataset_test, dict_users, dict_users_test = load_data(dataset_choice, num_users)


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test(net_client,net_server, validation_loader, device):
    net_client.eval()
    net_server.eval()
    correct = 0
    total = 0
    acc_list = []

    with torch.no_grad():
        len_batch = len(validation_loader)
        for batch_idx, (images, labels) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)
            fx_client = net_client(images).to(device)
            fx_server = net_server(fx_client)
            loss = criterion(fx_server,labels)
            acc = calculate_accuracy(fx_server,labels)
            acc_list.append(acc)

            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
    avg_acc = sum(acc_list) / len(acc_list)

    return avg_acc


def apply_roni(w_locals, w_glob, validation_loader, net_glob, device, roni_threshold):
    accepted_updates = []

    for w_local in w_locals:
        # 计算更新前的准确率
        original_accuracy = test(w_local,net_glob, validation_loader, device)

        # 应用客户端更新
        updated_net_glob = copy.deepcopy(net_glob)
        # updated_net_glob.load_state_dict(add_weights(w_glob, w_local))
        added = add_weights(w_glob, w_local)
        for key in added.keys():
            added[key] /= 2
        updated_net_glob.load_state_dict(added)

        # 计算更新后的准确率
        updated_accuracy = test(updated_net_glob, validation_loader, device)

        # 判断是否接受客户端更新
        if updated_accuracy - original_accuracy >= roni_threshold:
            accepted_updates.append(w_local)

    return accepted_updates


# def test(net, validation_loader, device):
#     net.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in validation_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = correct / total
#     return accuracy


# class Roni:
#     def __init__(self, train_server, evaluate_server,ell):
#         self.train_server = train_server
#         self.evaluate_server = evaluate_server
#         self.ell = ell

#     def apply_roni(self, fx_client, y, l_epoch_count, l_epoch, idx, len_batch,ell):
#         # 尝试在当前客户端上训练并评估
#         dfx_client = self.train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch)
#         acc, loss = self.evaluate_server(fx_client, y, idx, len_batch, ell, [])

#         # 判断是否通过RONI检测
#         if self.roni_check(acc, loss):
#             return dfx_client
#         else:
#             print(f"Client {idx} failed RONI check, skipping this client.")
#             return None

#     def roni_check(self, acc, loss):
#         # 在这里实现您的 RONI 检测方法
#         # 例如，您可以设置阈值并检查准确率和损失是否在合理范围内
#         acc_threshold = 0.5
#         loss_threshold = 2.0

#         return acc >= acc_threshold and loss <= loss_threshold

#     def run_test(self, idx, train_loader, test_loader, iter, selected_clients):
#         # 在这里实现您的 run_test 方法
#         fx_client, y, l_epoch_count, l_epoch, idx, len_batch = selected_clients
#         dfx_client = self.apply_roni(fx_client, y, l_epoch_count, l_epoch, idx, len_batch,self.ell)

#         if dfx_client is not None:
#             train_acc, train_loss = self.train_server(train_loader, iter, selected_clients)
#             test_acc, test_loss = self.evaluate_server(test_loader, iter, selected_clients)

#             passed = self.roni_check(train_acc, train_loss) and self.roni_check(test_acc, test_loss)
#         else:
#             passed = False
#             train_acc, train_loss, test_acc, test_loss = None, None, None, None

#         return passed, train_acc, train_loss, test_acc, test_loss


def subtract_weights(w1, w2):
    """
    计算 w1 和 w2 之间的差，即 w1 - w2。

    Args:
        w1 (OrderedDict): 权重字典 1
        w2 (OrderedDict): 权重字典 2

    Returns:
        OrderedDict: w1 和 w2 之间的权重差异
    """
    diff = OrderedDict()
    for key in w1.keys():
        diff[key] = w1[key] - w2[key]
    return diff


def add_weights(w1, w2):
    """
    对 w1 和 w2 进行加法操作，即 w1 + w2。

    Args:
        w1 (OrderedDict): 权重字典 1
        w2 (OrderedDict): 权重字典 2

    Returns:
        OrderedDict: w1 和 w2 相加的结果
    """
    added = OrderedDict()
    for key in w1.keys():
        added[key] = w1[key] + w2[key]
    return added

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

# =============================================================================
#                         Poisoning
# =============================================================================

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

# 定义攻击模式和目标标签
# attack_pattern = np.full((32, 32, 3), 255, dtype=np.uint8)
# target_label = 5

# poisoned_frac = 0.5
# poisoned_users_num = int(poisoned_frac * num_users)
# # label_mappings = [(4, 2), (1, 7), (3, 1), (2, 6), (6, 5)]
# label_mappings = generate_label_mappings(dataset_choice)
# poisoned_dict_users = poison_data(dataset_train, dataset_choice, dict_users, poisoned_users_num, label_mappings)
# # # poisoned_dict_users = poison_data_model(dataset_train, dataset_choice, dict_users, poisoned_users_num, attack_pattern, target_label)
# for poisoned_user_key in poisoned_dict_users:
#     print("被投毒的用户:", poisoned_user_key)

# # 如果使用的是 HAM10000 数据集，您可以直接使用 dataset_train.df 查看标签分布
# if dataset_choice == 'HAM10000':
#     print("标签反转后的target统计:")
#     print(dataset_train.df['target'].value_counts())
# elif dataset_choice == 'CIFAR10':
#     cifar10_df = cifar10_to_dataframe(dataset_train)
#     print("标签反转后的target统计:")
#     print(cifar10_df['target'].value_counts())

# ------------ Training And Testing  -----------------
net_glob_client.train()
# copy weights
w_glob_client = net_glob_client.state_dict()
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

total_time = 0.0  # 初始化总时间为0
best_clients_indices = None
krum_acc_test_collect = []
krum_loss_test_collect = []




for iter in range(epochs):
    start_time = time.time()  # 记录开始时间
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)  # ，replace=False表示不允许重复选择。
    w_locals_client = []  # 用于存储每个客户端训练后的本地模型参数。
    loss_avg_test_all_dict = {}
    acc_avg_test_all_dict = {}
    idx_local_client_dict = {}

    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        idx_local_client_dict[len(w_locals_client)] = idx
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter, selected_clients=best_clients_indices)

        # Update the dictionaries with the client's self.loss_avg_test_all and self.acc_avg_test_all
        loss_avg_test_all_dict[idx] = local.loss_avg_test_all
        acc_avg_test_all_dict[idx] = local.acc_avg_test_all


    print("idxs_users",idxs_users)
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

