#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 12:33
# @Author  : Andy_Arthur
# @File    : init_func.py
# @Software: win10
import argparse
import logging
import random

import numpy as np
import torch
from torch import nn

from Functions.Models import ResNet18_client_side, BasicBlock, ResNet18_server_side


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))

def init_logging(program_name):
    logging.basicConfig(filename=f'{program_name}.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

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

def parse_args():
    parser = argparse.ArgumentParser(description="SplitFed Learning")
    parser.add_argument('--num-users', type=int, default=10, help='number of users')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--frac', type=float, default=1, help='participation of clients')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--dataset', choices=['HAM10000', 'CIFAR10'], default='CIFAR10', help='dataset choice')
    return parser.parse_args()

# def main():
#     args = parse_args()
#
#     init_seeds(args.seed)
#     init_logging("SFLV1_ResNet18_base")
#
#     logging.info(f"---------{program}----------")
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     net_glob_client, net_glob_server = init_models(device, args.dataset)
#
#     # The rest of your code...
#
# if __name__ == '__main__':
#     main()
