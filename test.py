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

import torch
import crypten

crypten.init()

x = torch.tensor([1.0, 2.0, 3.0])
x_enc = crypten.cryptensor(x) # encrypt

x_dec = x_enc.get_plain_text() # decrypt

y_enc = crypten.cryptensor([2.0, 3.0, 4.0])
sum_xy = x_enc + y_enc # add encrypted tensors
sum_xy_dec = sum_xy.get_plain_text() # decrypt sum

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

# net_glob_client = ResNet18_client_side()
net_glob_client = ResNet18_client_side(BasicBlock, [2, 2, 2, 2]).to(device)
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(
        net_glob_client)  # to use the multiple GPUs; later we can change this to CPUs only

net_glob_client.to(device)
print(net_glob_client)

# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
# Model at server side
# Choose dataset: 'HAM10000' or 'CIFAR10'
dataset_choice = 'CIFAR10'
# Model at server side
if dataset_choice == 'HAM10000':
    num_classes = 7
    pool_size = 8
elif dataset_choice == 'CIFAR10':
    num_classes = 10
    pool_size = 4
else:
    raise ValueError('Invalid dataset choice.')

net_glob_server = ResNet18_server_side(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pool_size=pool_size)
# net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 7) #7 is my numbr of classes
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)  # to use the multiple GPUs

net_glob_server.to(device)
print(net_glob_server)

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
                    print("选择的客户端index:",selected_clients)
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
        return acc_avg_test_all,loss_avg_test_all
    else:
        return None,None

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                 idxs_test=None, is_attacker=None, batch_size=128):
        """
        :param idxs: idxs是一个表示该客户端用于训练的数据集的索引列表。在联邦学习中，原始数据集通常由多个客户端持有，每个客户端只能访问自己所持有的部分数据集。因此，为了让每个客户端只使用自己所持有的数据进行训练，需要将原始数据集划分成多个部分，每个部分由一个客户端持有，并通过idxs将该客户端用于训练的数据集的索引列表传递给Client类的构造函数。
        :param idxs_test:
        """
        # net_client_model:一个与客户端实例相关的神经网络模型。
        self.net_client_model = net_client_model
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

    def train(self, net, is_need_decode=False):
        # 解密全局模型参数
        if(is_need_decode):
            w_glob_decrypted = {k: v.get_plain_text().float() for k, v in self.net_client_model.items()}
            net.load_state_dict(w_glob_decrypted)
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

        for iter in range(self.local_ep):
            if self.is_attacker:
                dataset_split = self.ldr_train.dataset
                num_malicious_samples = len(dataset_split.idxs)
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

        # 对网络参数进行加密
        w_client = net.state_dict()
        w_client_encrypted = {k: crypten.cryptensor(v) for k, v in w_client.items()}

        return w_client_encrypted



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

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def add_malicious_samples(self, num_malicious_samples):
        for i in range(num_malicious_samples):
            # Randomly select an image to modify
            dataset_idx = random.choice(self.idxs)
            img = self.dataset.data[dataset_idx]

            # Check if the image has the shape (3, 32, 32)
            if img.shape == (3, 32, 32):
                img = np.transpose(img, (1, 2, 0))  # Change the shape to (32, 32, 3)

            # Convert the numpy array to a PIL Image
            img = Image.fromarray(img)

            # Apply a random modification to the image (e.g., flip, rotate, or change color)
            # In this example, we flip the image horizontally, rotate, and change the color
            modified_img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip the image horizontally
            modified_img = modified_img.rotate(random.randint(0, 360))  # Rotate the image randomly

            # Change the color by adjusting the brightness, contrast, and saturation
            brightness = ImageEnhance.Brightness(modified_img)
            modified_img = brightness.enhance(random.uniform(0.5, 1.5))

            contrast = ImageEnhance.Contrast(modified_img)
            modified_img = contrast.enhance(random.uniform(0.5, 1.5))

            saturation = ImageEnhance.Color(modified_img)
            modified_img = saturation.enhance(random.uniform(0.5, 1.5))

            # Convert the modified PIL Image back to a numpy array
            modified_img = np.array(modified_img)

            # Check if the original image has the shape (3, 32, 32)
            if self.dataset.data[dataset_idx].shape == (3, 32, 32):
                modified_img = np.transpose(modified_img, (2, 0, 1))  # Change the shape back to (3, 32, 32)

            # Update the image in the dataset
            self.dataset.data[dataset_idx] = modified_img

            # # Change the label of the modified image to a random class
            # random_label = random.randint(0, 9)  # Assuming 10 classes in the dataset
            # self.dataset.targets[dataset_idx] = random_label

class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

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
        dataset_train = SkinData(train, transform=train_transforms)
        dataset_test = SkinData(test, transform=test_transforms)

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

# =============================================================================
#                         Data loading
# =============================================================================
dataset_train, dataset_test, dict_users, dict_users_test = load_data(dataset_choice, num_users)

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

def FedAvg_SMPC(w_locals_encrypted):
    w_glob_encrypted = copy.deepcopy(w_locals_encrypted[0])

    for k in w_glob_encrypted.keys():
        for i in range(1, len(w_locals_encrypted)):
            w_glob_encrypted[k] += w_locals_encrypted[i][k]
        w_glob_encrypted[k] = w_glob_encrypted[k] / len(w_locals_encrypted)

    # w_glob = {}
    # for k in w_glob_encrypted.keys():
    #     w_glob[k] = w_glob_encrypted[k].get_plain_text().float()

    return w_glob_encrypted

is_need_decode = False
w_glob_encrypted = None

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
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx], is_attacker=None)
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