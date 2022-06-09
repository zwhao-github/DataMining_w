import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from collections import defaultdict
from math import sqrt
import os.path


# 划入相应簇
def point_avg(points):
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0
        for p in points:
            dim_sum += p[dimension]

        new_center.append(dim_sum / float(len(points)))

    return new_center


# 更新簇中心
def update_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


# 判断距离，确定簇标记
def assign_points(data_points, centers):
    assignments = []
    for point in data_points:
        shortest = float('inf')
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


# 计算距离
def distance(a, b):
    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


# 确定初始均值向量
def generate_k(data_set, k, n):
    random.seed(n)
    centers = []
    for _ in range(k):
        centers.append(list(np.array(data_set)[np.random.randint(0, np.array(data_set).shape[0])]))

    return centers


# K-means
def k_means(dataset, k, n=23):
    k_points = generate_k(dataset, k, n)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return zip(assignments, dataset)


# 读文件
data = pd.read_csv(os.path.dirname(__file__) + "\\R15.txt", sep='\t', header=None, names=["x", "y", "class"])
# print(data)
X = data[["x", "y"]].values
y = data["class"].values
# print(list(k_means(X, 15)))


# 实现
y_pre = []
for i, (cls, data) in enumerate(k_means(X, 15)):
    y_pre.append(cls)


# 可视化
plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.scatter(x=X[:, 0],y=X[:, 1],c=y)
plt.title("Before K Means Classificaion")
plt.subplot(1,2,2)
plt.scatter(x=X[:, 0] ,y=X[:, 1],c=y_pre)
plt.title("K means Classifcation")
plt.show()

# 存入文件
y_pre = np.array(y_pre)
y_pre[y_pre == 0] = 15
data1 = pd.DataFrame(y_pre)
data1.to_csv("pre2.txt", index=None, header=None)