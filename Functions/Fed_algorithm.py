import copy
from collections import OrderedDict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_with_grad_clipping(w_locals, clip_value=1):
    w_avg = copy.deepcopy(w_locals[0])
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k]

        w_avg[k] = torch.div(w_avg[k], len(w_locals))

        # 梯度裁剪
        w_avg[k] = torch.clamp(w_avg[k], -clip_value, clip_value)

    return w_avg



def FedMedian(w_locals):
    w_median = {}
    for k in w_locals[0].keys():
        w_median[k] = torch.median(torch.stack([w[k] for w in w_locals]), dim=0).values
    return w_median

def trimmed_and_thresholded_aggregation(weight_diffs, trim_ratio=0.1, threshold=0.01):
    """
    :param weight_diffs: 客户端权重差列表，每个元素是一个客户端的权重差
    :param trim_ratio: 用于权重修剪的比例,例如，当 trim_ratio=0.1 时，表示要剔除权重中绝对值最大的 10%。
    :param threshold: 用于阈值聚合的阈值,表示阈值聚合时的阈值。只有权重差的绝对值小于这个阈值的权重才会被聚合。
    :return:
    """

    # 对权重差进行修剪
    trimmed_weight_diffs = []
    for diff in weight_diffs:
        trimmed_diff = {}
        for key in diff.keys():
            # 然后，对于权重差中的每个键（层），代码将该键对应的权重差提取出来
            layer_diff = diff[key]
            layer_abs_diff = torch.abs(layer_diff)
            sorted_indices = torch.argsort(layer_abs_diff.view(-1), descending=True)
            trim_index = int(len(sorted_indices) * (1 - trim_ratio))
            indices_to_keep = sorted_indices[:trim_index]
            mask = torch.zeros_like(layer_diff, dtype=torch.bool)
            mask.view(-1)[indices_to_keep] = True
            trimmed_layer_diff = torch.where(mask, layer_diff, torch.zeros_like(layer_diff))
            trimmed_diff[key] = trimmed_layer_diff
        trimmed_weight_diffs.append(trimmed_diff)

    # 对修剪后的权重差进行阈值聚合
    # 初始化一个空字典 aggregated_weight_diff，用于存储阈值聚合后的权重差。
    aggregated_weight_diff = {}
    for i, weight_diff in enumerate(trimmed_weight_diffs):
        for k, v in weight_diff.items():
            if i == 0:
                # 对于第一个权重差，将其除以客户端数量并存储在 aggregated_weight_diff 中。
                aggregated_weight_diff[k] = v / float(len(trimmed_weight_diffs))
            else:
                # 对于其他权重差，如果其绝对值小于阈值 threshold，则将其除以客户端数量并累加到 aggregated_weight_diff 对应的键中。
                mask = torch.abs(v) < threshold
                aggregated_weight_diff[k] += torch.where(mask, v / float(len(trimmed_weight_diffs)), torch.zeros_like(v, dtype=torch.float))

    return aggregated_weight_diff




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


def dynamic_threshold_statistical(gradients_norms, std_multiplier=3):
    """
    动态计算阈值，使用平均值加减标准差的方法。
    :param gradients_norms: 客户端更新梯度范数列表。
    :param std_multiplier: 标准差倍数，用于确定阈值。
    :return: 动态计算出的阈值。
    """
    mean = np.mean(gradients_norms)
    std_dev = np.std(gradients_norms)
    threshold = mean + std_multiplier * std_dev
    return threshold




def dynamic_threshold_clustering(gradients_norms, n_clusters=2):
    """
    动态计算阈值，使用K-means聚类方法。
    :param gradients_norms: 客户端更新梯度范数列表。
    :param n_clusters: 聚类的数量，通常设置为2（正常更新和异常更新）。
    :return: 动态计算出的阈值。
    """
    # 将梯度范数转换为适用于K-means算法的格式。
    gradients_norms = np.array(gradients_norms).reshape(-1, 1)

    # 使用K-means聚类算法。
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradients_norms)

    # 获取聚类中心，并按降序排列。
    cluster_centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)

    # 假设异常更新的范数较大，选择距离聚类中心最远的异常更新作为阈值。
    threshold = (cluster_centers[0] + cluster_centers[1]) / 2
    return threshold


# def remove_anomalies(w_locals, w_glob, threshold):
#     """
#     基于梯度范数的异常检测方法
#     :param w_locals:客户端模型更新列表
#     :param w_glob:全局模型权重
#     :param threshold:阈值
#     :return:
#     """
#     # 检测异常客户端的索引。
#     anomaly_indices = detect_anomaly(w_locals, w_glob, threshold)
#     print("异常客户端索引:",anomaly_indices)
#     # 删除异常客户端的模型更新
#     w_locals = [w for idx, w in enumerate(w_locals) if idx not in anomaly_indices]
#     return w_locals

def remove_anomalies(w_locals, w_glob, threshold_method='statistical', std_multiplier=3, n_clusters=2):
    """
    基于梯度范数的异常检测方法
    :param w_locals: 客户端模型更新列表
    :param w_glob: 全局模型权重
    :param threshold_method: 选择动态阈值计算方法 ('statistical' 或 'clustering')
    :param std_multiplier: 标准差倍数，仅在 threshold_method='statistical' 时使用
    :param n_clusters: 聚类的数量，仅在 threshold_method='clustering' 时使用
    :return: 删除异常客户端的模型更新后的列表
    """
    # 计算客户端更新梯度范数列表。
    gradients_diffs = [subtract_weights(w_local, w_glob) for w_local in w_locals]
    # gradients_norms = [np.linalg.norm(np.concatenate([np.ravel(v) for v in diff.values()])) for diff in gradients_diffs]
    gradients_norms = [np.linalg.norm(np.concatenate([np.ravel(v.cpu()) for v in diff.values()])) for diff in
                       gradients_diffs]

    # 计算动态阈值。
    if threshold_method == 'statistical':
        threshold = dynamic_threshold_statistical(gradients_norms, std_multiplier)
    elif threshold_method == 'clustering':
        threshold = dynamic_threshold_clustering(gradients_norms, n_clusters)
    else:
        raise ValueError("Invalid threshold_method. Choose either 'statistical' or 'clustering'.")

    # 检测异常客户端的索引。
    anomaly_indices = detect_anomaly(w_locals, w_glob, threshold)
    print("异常客户端索引:", anomaly_indices)

    # 删除异常客户端的模型更新。
    w_locals = [w for idx, w in enumerate(w_locals) if idx not in anomaly_indices]
    return w_locals


def remove_anomalies_kmeans(w_locals, w_glob, n_clusters, remove_ratio):
    gradients_diffs = [subtract_weights(w_local, w_glob) for w_local in w_locals]
    gradients_list = []

    for diff in gradients_diffs:
        diff_list = [v.cpu().numpy().flatten() for v in diff.values()]
        gradients_list.append(np.concatenate(diff_list))

    scaler = StandardScaler()
    scaled_gradients = scaler.fit_transform(gradients_list)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scaled_gradients)
    cluster_labels = kmeans.labels_

    cluster_sizes = np.bincount(cluster_labels)
    smallest_cluster = np.argmin(cluster_sizes)

    remove_count = int(remove_ratio * len(w_locals))
    smallest_cluster_indices = np.argsort(np.linalg.norm(kmeans.transform(gradients_list), axis=1))[:remove_count]

    w_locals = [w for idx, w in enumerate(w_locals) if idx not in smallest_cluster_indices]

    return w_locals,smallest_cluster_indices


def detect_anomaly(w_locals, w_glob, threshold):
    gradients_diffs = [subtract_weights(w_local, w_glob) for w_local in w_locals]
    gradients_norms = [np.linalg.norm(np.concatenate([np.ravel(v.cpu()) for v in diff.values()])) for diff in gradients_diffs]

    # 将梯度范数转换为张量
    gradients_norms_tensor = torch.tensor(gradients_norms)

    # 计算平均值和标准差
    mean = torch.mean(gradients_norms_tensor)
    std = torch.std(gradients_norms_tensor)

    # 计算异常阈值
    threshold = mean + threshold * std

    # 检测异常客户端
    anomaly_indices = [idx for idx, norm in enumerate(gradients_norms) if norm > threshold]

    return anomaly_indices


# def detect_anomaly(w_locals, w_glob_client,threshold):
#     grad_norms = [] # 用于存储所有客户端模型更新的梯度范数。
#     # 计算所有客户端模型更新的梯度范数（与全局模型之间的差异）
#     for idx, w in enumerate(w_locals):
#         # 计算当前客户端模型更新与全局模型权重之间的差异，即梯度范数。
#         diff = subtract_weights(w, w_glob_client)
#         diff_list = [v.flatten() for v in diff.values()]
#         diff_tensor = torch.cat(diff_list)
#         grad_norm = torch.norm(diff_tensor, p=2)
#         grad_norms.append(grad_norm.item())
#
#     # 计算梯度范数的标准差和平均值
#     std_dev = np.std(grad_norms)
#     mean_grad_norm = np.mean(grad_norms)
#     # 确定异常客户端（梯度范数大于平均值加减某个阈值倍的标准差）
#     anomaly_indices = [idx for idx, norm in enumerate(grad_norms) if abs(norm - mean_grad_norm) > threshold * std_dev]
#
#     return anomaly_indices


def krum_aggregation(weight_dicts, num_to_select):
    # ...（weights_to_array、array_to_weights和pairwise_distances函数保持不变）
    def weights_to_array(weight_dict):
        """
        将权重字典转换为 numpy 数组。
        :param weight_dict: 权重字典，其中每个值都是一个 torch.Tensor 张量。
        :return: 一个 numpy 数组，其中包含了所有权重张量的扁平化数组。
        """
        # 初始化 weight_list 列表，用于存储 weight_dict 中每个权重张量的扁平化数组。
        weight_list = []
        for key in weight_dict:
            # 将权重张量转换为 numpy 数组，并使用 flatten 函数将其扁平化
            weight_list.append(weight_dict[key].cpu().numpy().flatten())
        # 将 weight_list 中的数组连接成一个 numpy 数组，并返回该数组
        return np.concatenate(weight_list)

    def array_to_weights(array, weight_dict_template):
        """
        从 numpy 数组中提取权重张量，并将它们保存到一个新的权重字典中。
        :param array: 包含所有权重张量的扁平化 numpy 数组。
        :param weight_dict_template: 一个权重字典模板，其中包含了所有权重张量的形状。
        :return: 一个新的权重字典，其中包含了从数组中提取的权重张量。
        """
        # 初始化一个新的有序字典 new_weight_dict，用于存储从 numpy 数组中提取的权重张量
        new_weight_dict = OrderedDict()
        # 初始化一个索引变量 idx，用于跟踪从数组中提取权重的位置
        idx = 0
        # 遍历权重字典模板中的每个键 key
        for key in weight_dict_template:
            # 获取权重张量的大小 size
            size = weight_dict_template[key].numel()
            # 从 numpy 数组中提取一个与权重张量相同大小的一维切片，并使用 reshape 函数重新塑形
            # 将 numpy 数组转换为 torch.Tensor 张量，并将其存储到新的有序字典 new_weight_dict 中
            new_weight_dict[key] = torch.from_numpy(array[idx:idx + size].reshape(weight_dict_template[key].shape))
            # 更新索引变量 idx 的值，跳过已经提取的权重张量
            idx += size
        # 返回新的有序字典 new_weight_dict
        return new_weight_dict

    def pairwise_distances(weight_updates):
        # 获取客户端数量
        n_clients = len(weight_updates)
        # 初始化一个距离矩阵，矩阵大小为 n_clients * n_clients
        distances = np.zeros((n_clients, n_clients))

        # 遍历所有的客户端对
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                # 计算 i 和 j 客户端之间的欧氏距离
                dist = np.linalg.norm(weight_updates[i] - weight_updates[j])
                # 在距离矩阵中记录距离
                distances[i, j] = dist
                distances[j, i] = dist

        # 返回距离矩阵
        return distances

    weight_arrays = [weights_to_array(weight_dict) for weight_dict in weight_dicts]
    n_clients = len(weight_arrays)
    distances = pairwise_distances(weight_arrays)

    krum_scores = []
    for i in range(n_clients):
        sorted_distances = np.sort(distances[i])
        # 计算当前客户端的Krum分数，即距离最大的n_clients - num_to_select - 1个客户端之间的距离总和。
        krum_score = np.sum(sorted_distances[-(n_clients - num_to_select - 1):])
        krum_scores.append(krum_score)

    # 使用np.argpartition函数找到具有最低Krum分数的num_to_select个客户端的索引。
    best_clients_indices = np.argpartition(krum_scores, num_to_select)[:num_to_select]

    # 从权重数组中选择最佳客户端的权重。
    selected_weight_arrays = [weight_arrays[i] for i in best_clients_indices]
    # 计算选定客户端的权重数组的平均值，得到聚合后的权重数组。
    aggregated_weight_array = np.mean(selected_weight_arrays, axis=0)

    # 使用array_to_weights函数将聚合后的权重数组转换回权重字典。
    aggregated_weights = array_to_weights(aggregated_weight_array, weight_dicts[0])

    # 返回聚合后的权重字典和最好的客户端索引列表。
    return aggregated_weights, best_clients_indices

# # 定义一个函数，输入参数为用户的梯度更新、用户的数量、拜占庭错误的数量、距离矩阵（可选）、是否返回索引（可选）、是否调试（可选）
# def krum(users_grads, users_count, corrupted_count, distances=None,return_index=False, debug=False):
#     # 如果不返回索引，那么检查用户的数量是否满足至少比拜占庭错误多一倍加一，否则报错
#     if not return_index:
#         assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
#     # 计算非恶意用户的数量
#     num_selected = users_count - corrupted_count
#     k = num_selected - 2
#     # 初始化最小误差为一个很大的数
#     minimal_error = 1e20
#     # 初始化最小误差对应的用户索引为-1
#     selected_index = -1
#
#     # 如果没有给定距离矩阵，那么调用_krum_create_distances函数创建一个距离矩阵，存储每个用户之间的欧氏距离
#     if distances is None:
#         distances = _krum_create_distances(users_grads)
#     selected_index, k_closest_indices = _krum_select_k_closest(distances, k)
#     print("选择的index：",selected_index)
#     print("离k最近的k个索引",k_closest_indices)
#     # 计算k个最近用户的梯度更新的平均值
#     k_closest_grads = [users_grads[i] for i in k_closest_indices]
#     average_grad = _calculate_average_gradient(k_closest_grads)
#     # 如果返回索引，那么返回最小误差对应的索引；否则返回最小误差对应的梯度更新
#     if return_index:
#         return selected_index
#     else:
#         return average_grad,k_closest_indices
#
#
# def _krum_select_k_closest(distances, k):
#     """
#
#     :param distances:输入参数 distances 是一个包含每个客户端与其他客户端的距离的列表，其中 distances[i][j] 表示第 i 个客户端与第 j 个客户端之间的距离。
#     :param k:k 是一个超参数，表示选出距离第 i 个客户端最近的 k 个客户端进行评估
#     :return:
#     """
#     num_users = len(distances)
#     # 初始化 selected_index 和 min_sum_distance。selected_index 表示最可信的客户端的索引，初始值为 -1
#     selected_index = -1
#     # min_sum_distance 初始值为正无穷大，用于存储当前选出的客户端集合中距离之和最小的值。
#     min_sum_distance = float('inf')
#     selected_k_closest_indices = []
#     for i in range(num_users):
#         """
#         对于每个客户端 i，计算它与其他客户端的距离并选出距离最近的 k 个客户端，然后计算这 k 个客户端与客户端 i 之间的距离之和。这里使用了 NumPy 库中的 np.sum 和 np.sort 函数，np.sort(distances[i]) 对第 i 行距离列表进行排序，[1:k+1] 表示选出排序后的第 2 到第 k+1 个元素，即距离第 i 个客户端最近的 k 个客户端。
#         """
#         sorted_indices = np.argsort(distances[i])
#         sum_distance = np.sum(np.sort(distances[i])[1:k + 1])
#         k_closest_indices = sorted_indices[:k + 1]
#
#         if sum_distance < min_sum_distance:
#             min_sum_distance = sum_distance
#             selected_index = i
#             selected_k_closest_indices = k_closest_indices
#
#     return selected_index, selected_k_closest_indices
#
# def _krum_create_distances(users_grads:list):
#     """
#
#     :param users_grads: list类型，里面是有序字典，存储w权重参数
#     :return:
#     """
#     user_nums = len(users_grads)
#     distances = np.zeros((user_nums,user_nums))
#     for user_index,user_grad in enumerate(users_grads):
#         # user_grad是一个order_dict
#         for other_user_index,other_grad in enumerate(users_grads):
#             if other_user_index == user_index:
#                 continue
#             else:
#                 # dis_ordered_dict = OrderedDict()
#                 d = 0
#                 for layer in user_grad.keys():
#                     d += torch.norm(user_grad[layer].float()-other_grad[layer].float(), p=2)
#                 distances[user_index][other_user_index] = distances[other_user_index][user_index] = d
#     return distances
#
#     # for user in users_grads.keys():
#     #     distances[user] = {}
#     #     for other_user in users_grads.keys():
#     #         if user == other_user:
#     #             continue
#     #         distances[user][other_user] = np.linalg.norm(users_grads[user] - users_grads[other_user])
#     # return distances
#
# # 新增一个函数来计算k个最近用户梯度更新的平均值
# def _calculate_average_gradient(k_closest_grads):
#     k = len(k_closest_grads)
#     average_grad = {}
#
#     for layer in k_closest_grads[0].keys():
#         layer_sum = 0
#         for i in range(k):
#             layer_sum += k_closest_grads[i][layer].float()
#         average_grad[layer] = layer_sum / k
#
#     return average_grad

# def krum_aggregation(weight_dicts, n_malicious_clients):
#     def weights_to_array(weight_dict):
#         weight_list = []
#         for key in weight_dict:
#             weight_list.append(weight_dict[key].cpu().numpy().flatten())
#         return np.concatenate(weight_list)
#
#     def array_to_weights(array, weight_dict_template):
#         new_weight_dict = OrderedDict()
#         idx = 0
#         for key in weight_dict_template:
#             size = weight_dict_template[key].numel()
#             new_weight_dict[key] = torch.from_numpy(array[idx:idx+size].reshape(weight_dict_template[key].shape))
#             idx += size
#         return new_weight_dict
#
#     def pairwise_distances(weight_updates):
#         n_clients = len(weight_updates)
#         distances = np.zeros((n_clients, n_clients))
#
#         for i in range(n_clients):
#             for j in range(i + 1, n_clients):
#                 dist = np.linalg.norm(weight_updates[i] - weight_updates[j])
#                 distances[i, j] = dist
#                 distances[j, i] = dist
#
#         return distances
#
#     weight_arrays = [weights_to_array(weight_dict) for weight_dict in weight_dicts]
#     n_clients = len(weight_arrays)
#     distances = pairwise_distances(weight_arrays)
#
#     krum_scores = []
#     for i in range(n_clients):
#         sorted_distances = np.sort(distances[i])
#         krum_score = np.sum(sorted_distances[:n_clients - n_malicious_clients - 1])
#         krum_scores.append((i, krum_score))
#
#     # Select the client with the lowest Krum score
#     selected_client_index = min(krum_scores, key=lambda x: x[1])[0]
#
#     return array_to_weights(weight_arrays[selected_client_index], weight_dicts[0])


# def krum_aggregation(weight_dicts, num_to_select):
#     def weights_to_array(weight_dict):
#         """
#         将权重字典转换为 numpy 数组。
#         :param weight_dict: 权重字典，其中每个值都是一个 torch.Tensor 张量。
#         :return: 一个 numpy 数组，其中包含了所有权重张量的扁平化数组。
#         """
#         # 初始化 weight_list 列表，用于存储 weight_dict 中每个权重张量的扁平化数组。
#         weight_list = []
#         for key in weight_dict:
#             # 将权重张量转换为 numpy 数组，并使用 flatten 函数将其扁平化
#             weight_list.append(weight_dict[key].cpu().numpy().flatten())
#         # 将 weight_list 中的数组连接成一个 numpy 数组，并返回该数组
#         return np.concatenate(weight_list)
#
#     def array_to_weights(array, weight_dict_template):
#         """
#         从 numpy 数组中提取权重张量，并将它们保存到一个新的权重字典中。
#         :param array: 包含所有权重张量的扁平化 numpy 数组。
#         :param weight_dict_template: 一个权重字典模板，其中包含了所有权重张量的形状。
#         :return: 一个新的权重字典，其中包含了从数组中提取的权重张量。
#         """
#         # 初始化一个新的有序字典 new_weight_dict，用于存储从 numpy 数组中提取的权重张量
#         new_weight_dict = OrderedDict()
#         # 初始化一个索引变量 idx，用于跟踪从数组中提取权重的位置
#         idx = 0
#         # 遍历权重字典模板中的每个键 key
#         for key in weight_dict_template:
#             # 获取权重张量的大小 size
#             size = weight_dict_template[key].numel()
#             # 从 numpy 数组中提取一个与权重张量相同大小的一维切片，并使用 reshape 函数重新塑形
#             # 将 numpy 数组转换为 torch.Tensor 张量，并将其存储到新的有序字典 new_weight_dict 中
#             new_weight_dict[key] = torch.from_numpy(array[idx:idx + size].reshape(weight_dict_template[key].shape))
#             # 更新索引变量 idx 的值，跳过已经提取的权重张量
#             idx += size
#         # 返回新的有序字典 new_weight_dict
#         return new_weight_dict
#
#     def pairwise_distances(weight_updates):
#         # 获取客户端数量
#         n_clients = len(weight_updates)
#         # 初始化一个距离矩阵，矩阵大小为 n_clients * n_clients
#         distances = np.zeros((n_clients, n_clients))
#
#         # 遍历所有的客户端对
#         for i in range(n_clients):
#             for j in range(i + 1, n_clients):
#                 # 计算 i 和 j 客户端之间的欧氏距离
#                 dist = np.linalg.norm(weight_updates[i] - weight_updates[j])
#                 # 在距离矩阵中记录距离
#                 distances[i, j] = dist
#                 distances[j, i] = dist
#
#         # 返回距离矩阵
#         return distances
#
#     weight_arrays = [weights_to_array(weight_dict) for weight_dict in weight_dicts]
#     n_clients = len(weight_arrays)
#     distances = pairwise_distances(weight_arrays)
#
#     krum_scores = []
#     for i in range(n_clients):
#         sorted_distances = np.sort(distances[i])
#         krum_score = np.sum(sorted_distances[:n_clients - num_to_select - 1])
#         krum_scores.append(krum_score)
#
#     # 找到最佳客户端的索引
#     best_clients_indices = np.argpartition(krum_scores, num_to_select)[:num_to_select]
#
#     # 计算选中客户端权重的平均值
#     selected_weight_arrays = [weight_arrays[i] for i in best_clients_indices]
#     aggregated_weight_array = np.mean(selected_weight_arrays, axis=0)
#
#     aggregated_weights = array_to_weights(aggregated_weight_array, weight_dicts[0])
#
#     # 返回聚合后的权重字典和最好的客户端索引列表。
#     return aggregated_weights, best_clients_indices
