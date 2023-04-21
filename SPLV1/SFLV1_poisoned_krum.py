#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

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

import random
import numpy as np
import os


import matplotlib

from Functions.Data_proccess_func import SkinData, dataset_iid, DatasetSplit
from Functions.Fed_algorithm import FedAvg, krum
from Functions.Models import ResNet18_client_side, ResNet18_server_side, Baseblock
from Functions.train_test_func import calculate_accuracy, prGreen, prRed
from SplitFed_Security.Functions.class_flipping_df_method import replace_label1_with_label2_on_df, \
    random_select_poisoning_users

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

#===================================================================
program = "SFLV1 ResNet18 on HAM10000"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#===================================================================
# No. of users
num_users = 10
epochs = 100
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.0001

#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
net_glob_client = ResNet18_client_side()
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)

net_glob_client.to(device)
print(net_glob_client)

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side
net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], 7) #7 is my numbr of classes
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs

net_glob_server.to(device)
print(net_glob_server)

#===================================================================================
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

#====================================================================================================
#                                  Server Side Program
#====================================================================================================
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

#client idx collector
idx_collect = []    # 初始化一个空列表，用于收集选择的客户端的索引。
l_epoch_check = False   # 初始化一个布尔变量，用于指示是否进行了本地训练轮次的检查。
fed_check = False   # 初始化一个布尔变量，用于指示是否完成了联邦学习。
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(num_users)]  # 该列表包含了每个客户端的初始模型。
net_server = copy.deepcopy(net_model_server[0]).to(device)  # 初始化为net_model_server的第一个元素的深拷贝，并将其移到GPU上。
#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

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
    global k_closest_indices

    # net_server是全局模型，返回指定索引的本地模型
    net_server = copy.deepcopy(net_model_server[idx]).to(
        device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr)

    # 1.train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    # ---------forward prop-------------
    # 将 fx_client 作为输入传递到全局模型 net_server 中，然后返回模型的预测输出 fx_server
    fx_server = net_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    loss.backward()
    # 由于我们需要在全局模型更新之前将 fx_client 更新到最新的版本，因此我们使用 clone().detach() 函数来创建一个新的 dfx_client 张量，它具有相同的值但不会被计算图所记录。
    dfx_client = fx_client.grad.clone().detach()
    # 更新 optimizer_server 以更新全局模型参数
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Update the server-side model for the current batch
    # 将更新后的全局模型 net_server 复制到全局模型列表 net_model_server 中指定的索引位置。
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

        # copy the last trained model in the batch，我们使用最后一个batch
        # 的状态字典复制到一个新的字典中，以便我们可以将其发送到参与者，从而启动下一轮的联邦学习。注意，w_server 中包含的参数是最新一轮训练的参数，因此每个参与者将从这些参数开始训练它们的本地模型。
        w_server = net_server.state_dict()

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:
            # # 标记已经完成本地epoch
            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
            # We store the state of the net_glob_server()
            # w_server 是全局模型中最新的训练参数，w_locals_server 是用于存储每个参与者的最后一轮训练参数的列表。因此，当本地epoch完成时，将 w_server 添加到 w_locals_server 中，以便之后将其发送到联邦平均服务器。
            w_locals_server.append(copy.deepcopy(w_server))

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
                print("已经训练的客户端:"+str(idx_collect))

        # This is for federation process--------------------
        if len(idx_collect) == num_users * frac:
            # 如果客户端编号列表的长度等于客户端总数，说明所有客户端的训练结果都已经到达服务器了。
            # 这里不对，是选择的客户端总数
            fed_check = True  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display

            w_glob_server,k_closest_indices = krum(w_locals_server,num_users,poisoned_users_num)
            # w_glob_server = FedAvg(w_locals_server)  # 使用联邦平均算法更新全局模型，将所有客户端的本地模型参数传入该函数中。

            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)  # 将更新后的全局模型参数加载到服务器端的模型中。
            net_model_server = [net_glob_server for i in
                                range(num_users)]  # 创建一个长度为客户端数量的列表，每个元素都是更新后的全局模型。这个列表用于向每个客户端分发全局模型参数。

            w_locals_server = []  # # 清空本地模型参数列表
            idx_collect = []  # 清空客户端编号列表

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)  # 计算所有客户端训练结果的平均准确率和损失
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client


def evaluate_server(fx_client, y, idx, len_batch, ell,idxs_users):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()

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

                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if federation is happened----------
            if fed_check:
                # 检查是否需要进行联邦学习服务器端处理。
                """
                这段代码的目的是在联邦学习过程的服务器端记录和输出每轮训练和测试的平均损失和准确率。
                """
                fed_check = False   # 将fed_check设置为False，表示本轮服务器端处理已完成。
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")

                # 获取 k_closest_indices 在 idxs_users 中的索引
                selected_indices_in_idxs_users = [idxs_users.index(idx) for idx in k_closest_indices]
                print("krum选中的用户索引",selected_indices_in_idxs_users)
                # 仅考虑 Krum 算法选中的客户端的准确率和损失
                acc_test_collect_user_selected = [acc_test_collect_user[i] for i in selected_indices_in_idxs_users]
                loss_test_collect_user_selected = [loss_test_collect_user[i] for i in selected_indices_in_idxs_users]

                acc_avg_selected_user = sum(acc_test_collect_user_selected) / len(acc_test_collect_user_selected)
                loss_avg_selected_user = sum(loss_test_collect_user_selected) / len(loss_test_collect_user_selected)
                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_selected_user)
                acc_test_collect.append(acc_avg_selected_user)
                # 清空每个客户端的测试准确率和测试损失列表，为下一轮训练做准备。
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_selected_user,
                                                                                         loss_avg_selected_user))
                print("==========================================================")

                # print("====================== SERVER V1==========================")
                # print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                #                                                                           loss_avg_all_user_train))
                # print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                #                                                                          loss_avg_all_user))
                # print("==========================================================")

    return

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
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256,
                                    shuffle=True)  # 一个PyTorch数据集，表示客户端可用于训练的数据
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256, shuffle=True)

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
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)

                # --------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()

            # prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return net.state_dict()

    def evaluate(self, net, ell,idxs_users):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)

                # Sending activations to server
                evaluate_server(fx, labels, self.idx, len_batch, ell,idxs_users)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))

        return

#=============================================================================
#                         Data loading
#=============================================================================
df = pd.read_csv('../data/HAM10000_metadata.csv')
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



#print("path---------------------------------------", imageid_path.get)
# 将图像id映射为图像文件的路径，并将其存储在数据集中的path列中。
df['path'] = df['image_id'].map(imageid_path.get)
# 将诊断编码映射为对应的分类名称，并将其存储在数据集中的cell_type列中。
df['cell_type'] = df['dx'].map(lesion_type.get)
# 将分类名称转换为数字编码，并将其存储在数据集中的target列中。这里使用了.
# 可以将字符串类型的分类变量转换为数字编码，其中不同的分类名称对应不同的数字编码。
df['target'] = pd.Categorical(df['cell_type']).codes
print(df['cell_type'].value_counts())
print(df['target'].value_counts())



#=============================================================================
# Train-test split
train, test = train_test_split(df, test_size = 0.2)

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

# =============================================================================
#                         Poisoning
# =============================================================================
poisoned_users_num = int(0.4*num_users)
poisoned_dict_users = random_select_poisoning_users(dict_users,poisoned_users_num)
replace_label1_with_label2_on_df(dataset_train.df,4,2,poisoned_dict_users)

for poisoned_user_key in poisoned_dict_users:
    print("被投毒的用户:",poisoned_user_key)

print("标签反转后的target统计:")
print(dataset_train.df['target'].value_counts())

# ------------ Training And Testing  -----------------
net_glob_client.train()
# copy weights
w_glob_client = net_glob_client.state_dict()
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = list(np.random.choice(range(num_users), m, replace=False))
    print("选中的用户顺序",idxs_users)
    # 初始化一个列表，用于存储每个选定客户端训练后的本地模型权重。
    w_locals_client = []

    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        # Training ------------------
        w_client =\
            local.train(net=copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))

        # Testing -------------------
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter,idxs_users=idxs_users)

    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    # w_glob_client = FedAvg(w_locals_client)
    w_glob_client,_ = krum(w_locals_client,num_users,poisoned_users_num)
    # Update client-side global model
    net_glob_client.load_state_dict(w_glob_client)

# ===================================================================================

print("Training and Evaluation completed!")

# ===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect) + 1)]
df = DataFrame({'round': round_process, 'acc_train': acc_train_collect, 'acc_test': acc_test_collect,
                'loss_train':loss_train_collect,'loss_test':loss_test_collect})
file_name = program + ".xlsx"
df.to_excel(file_name, sheet_name="poisoned_v1_test", index=False)

# =============================================================================
#                         Program Completed
# =============================================================================