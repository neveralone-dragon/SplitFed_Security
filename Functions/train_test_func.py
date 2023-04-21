import copy

import torch

from Functions.Fed_algorithm import FedAvg


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

def calculate_recall(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    true_positives = (predicted == labels).float().sum()
    actual_positives = (labels == labels).float().sum()
    recall = true_positives / actual_positives
    return recall * 100