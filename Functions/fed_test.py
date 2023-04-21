#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 17:41
# @Author  : Andy_Arthur
# @File    : fed_test.py
# @Software: win10
import pickle

from SplitFed_Security.Functions.Fed_algorithm import krum

# 从文件中读取OrderedDict对象
with open(r'D:\codes\DeepLearning\SplitFed_Security\w_locals_client.pickle', 'rb') as f:
    users_grads = pickle.load(f)

users_count = len(users_grads)
corrupted_count = 4

new_users_count = krum(users_grads,users_count,corrupted_count)
print(new_users_count)