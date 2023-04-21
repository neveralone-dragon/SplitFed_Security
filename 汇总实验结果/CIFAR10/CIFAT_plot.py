import pandas as pd
import os
import matplotlib.pyplot as plt



# 获取当前文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义文件名列表
filenames = ["SFLV1_label_random_0.1_CIFAR10.xlsx",
             "SFLV1_label_random_0.2_CIFAR10.xlsx",
             "SFLV1_label_random_0.3_CIFAR10.xlsx",
             "SFLV1_label_random_0.4_CIFAR10.xlsx",
             "SFLV1_label_random_0.5_CIFAR10.xlsx",
             "SFLV1_baseline_CIFAR10.xlsx"]
filenames = ["SFLV1_label_flipping_0.1_CIFAR10.xlsx",
             "SFLV1_label_flipping_0.2_CIFAR10.xlsx",
             "SFLV1_label_flipping_0.3_CIFAR10.xlsx",
             "SFLV1_label_flipping_0.4_CIFAR10.xlsx",
             "SFLV1_label_flipping_0.5_CIFAR10.xlsx",
             "SFLV1_baseline_CIFAR10.xlsx"]
filenames = ["SFLV1_change_picture_0.1_CIFAR10.xlsx",
             "SFLV1_change_picture_0.2_CIFAR10.xlsx",
             "SFLV1_change_picture_0.3_CIFAR10.xlsx",
             "SFLV1_change_picture_0.4_CIFAR10.xlsx",
             "SFLV1_change_picture_0.5_CIFAR10.xlsx",
             "SFLV1_baseline_CIFAR10.xlsx"]

label_names = ["SFLV1_label_flipping_0.1",
             "SFLV1_label_flipping_0.2",
             "SFLV1_label_flipping_0.3_",
             "SFLV1_label_flipping_0.4_",
             "SFLV1_label_flipping_0.5_",
             "SFLV1_baseline_CIFAR10"]



# 颜色列表
colors = ["red", "blue", "green", "orange", "purple", "gray"]

# 读取每个文件的acc_test数据
for i, filename in enumerate(filenames):
    # 构建文件的绝对路径
    file_path = os.path.join(current_dir, filename)

    df = pd.read_excel(file_path)
    acc_test = df['acc_test'].tolist()

    # 获取文件名部分并绘制折线图
    label_name = os.path.splitext(filename)[0]
    plt.plot(range(1, len(acc_test) + 1), acc_test, color=colors[i], label=label_names[i])

# 添加标题和标签
# plt.title("SFLV1_poison_data_and_model")
plt.title("SFLV1_label_flipping_random")
plt.xlabel("round")
plt.ylabel("acc_test")

# 添加图例
plt.legend()

# 显示图形
plt.show()

