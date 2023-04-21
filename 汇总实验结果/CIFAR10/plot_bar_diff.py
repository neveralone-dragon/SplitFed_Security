import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 文件名列表
file_names = [
    "SFLV1_label_random_0.1_CIFAR10.xlsx",
    "SFLV1_label_random_0.2_CIFAR10.xlsx",
    "SFLV1_label_random_0.3_CIFAR10.xlsx",
    "SFLV1_label_random_0.4_CIFAR10.xlsx",
    "SFLV1_label_random_0.5_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.1_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.2_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.3_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.4_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.5_CIFAR10.xlsx",
    "SFLV1_change_picture_0.1_CIFAR10.xlsx",
    "SFLV1_change_picture_0.2_CIFAR10.xlsx",
    "SFLV1_change_picture_0.3_CIFAR10.xlsx",
    "SFLV1_change_picture_0.4_CIFAR10.xlsx",
    "SFLV1_change_picture_0.5_CIFAR10.xlsx",
]

# 读取文件内容并存储到字典中
data = {}
for file_name in file_names:
    data[file_name] = pd.read_excel(file_name)

# 提取所需数据
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
test_acc = []
for ratio in ratios:
    for prefix in ["SFLV1_label_random_", "SFLV1_label_flipping_", "SFLV1_change_picture_"]:
        file_name = f"{prefix}{ratio}_CIFAR10.xlsx"
        test_acc.append(data[file_name]["loss_test"].iloc[-1])  # 提取最后一行的acc_test值

# 绘制柱状图
bar_width = 0.25
x = np.arange(len(ratios))
plt.figure(figsize=(10, 6))

plt.bar(x - bar_width, test_acc[0::3], width=bar_width, label="label_flipping")
plt.bar(x, test_acc[1::3], width=bar_width, label="poison_data_and_label")
plt.bar(x + bar_width, test_acc[2::3], width=bar_width, label="Change Picture")

plt.xlabel("Ratios")
plt.ylabel("Test Loss")
plt.xticks(x, ratios)
plt.legend()
plt.title("Test Loss by Ratio and Attack Type")
plt.show()
