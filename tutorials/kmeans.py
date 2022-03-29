import random
import numpy as np


# 初始化簇心
def get_init_centers(raw_data, k):
    return random.sample(raw_data, k)


# 计算距离
def cal_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))  # 欧氏距离


# 将各点分配到最近的点, 并计算MSE
def get_cluster_with_mse(raw_data, centers):
    distance_sum = 0.0
    cluster = {}
    for item in raw_data:
        flag = -1
        min_dis = float('inf')
        for i, center_point in enumerate(centers):
            dis = cal_distance(item, center_point)
            if dis < min_dis:
                flag = i
                min_dis = dis
        if flag not in cluster:
            cluster[flag] = []
        cluster[flag].append(item)
        distance_sum += min_dis ** 2
    return cluster, distance_sum / (len(raw_data) - len(centers))


# 计算各簇的中心点，获取新簇心
def get_new_centers(cluster):
    center_points = []
    for key in cluster.keys():
        center_points.append(np.mean(cluster[key], axis=0))  # axis=0，计算每个维度的平均值
    return center_points


# K means主方法
def k_means(raw_data, k, mse_limit, early_stopping):
    old_centers = get_init_centers(raw_data, k)
    old_cluster, old_mse = get_cluster_with_mse(raw_data, old_centers)
    count = 0
    new_cluster, new_mse = None, 0
    while np.abs(old_mse - new_mse) > mse_limit and count < early_stopping : 
        old_mse = new_mse
        new_center = get_new_centers(old_cluster)
        # print(new_center)
        new_cluster, new_mse = get_cluster_with_mse(raw_data, new_center)  
        count += 1
        print('mse:', np.abs(new_mse), 'Update times:', count)
    return new_cluster
