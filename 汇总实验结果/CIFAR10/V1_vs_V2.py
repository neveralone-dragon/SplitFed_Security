import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 文件名列表
file_names = [
    "SFLV1_label_flipping_0.1_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.2_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.3_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.4_CIFAR10.xlsx",
    "SFLV1_label_flipping_0.5_CIFAR10.xlsx",
    "SFLV2_label_flipping_0.1_CIFAR10.xlsx",
    "SFLV2_label_flipping_0.2_CIFAR10.xlsx",
    "SFLV2_label_flipping_0.3_CIFAR10.xlsx",
    "SFLV2_label_flipping_0.4_CIFAR10.xlsx",
    "SFLV2_label_flipping_0.5_CIFAR10.xlsx",
]

# 读取文件内容并存储到字典中
data = {}
for file_name in file_names:
    data[file_name] = pd.read_excel(file_name)

# 提取所需数据
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
test_acc_sflv1 = []
test_acc_sflv2 = []

for ratio in ratios:
    file_name_sflv1 = f"SFLV1_label_flipping_{ratio}_CIFAR10.xlsx"
    file_name_sflv2 = f"SFLV2_label_flipping_{ratio}_CIFAR10.xlsx"
    test_acc_sflv1.append(data[file_name_sflv1]["acc_test"].iloc[-1])  # 提取SFLV1最后一行的acc_test值
    test_acc_sflv2.append(data[file_name_sflv2]["acc_test"].iloc[-1])  # 提取SFLV2最后一行的acc_test值

# 绘制柱状图
bar_width = 0.25
x = np.arange(len(ratios))
plt.figure(figsize=(10, 6))

plt.bar(x - bar_width / 2, test_acc_sflv1, width=bar_width, label="SFLV1")
plt.bar(x + bar_width / 2, test_acc_sflv2, width=bar_width, label="SFLV2")

plt.xlabel("Ratios")
plt.ylabel("Test Accuracy")
plt.xticks(x, ratios)
plt.legend()
plt.title("Test Accuracy Comparison between SFLV1 and SFLV2 with poison data and label")
plt.show()
