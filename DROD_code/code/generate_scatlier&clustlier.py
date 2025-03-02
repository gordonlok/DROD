# generate outliers for the dataset
# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 22:37
# @Author  : Sheng
# @File    : generateOutliers.py
# @Software: PyCharm
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

CWD = os.path.dirname(os.getcwd())  # current working directory to save results to

matplotlib.use("Qt5Agg")

filestr = ["zelnik3"]
for str1 in filestr:
    data_path = os.path.join(CWD, "data/generate/base/{}.csv".format(str1))
    # data_path = os.path.join(CWD, "data/syntdata/clusteredscattered/{}_outlier_5_1.csv".format(str1))
    df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
    data = df.values
    dataset = data[:, :-1]
    label = data[:, -1]



    # "归一化（0，1）"
    scaler = MinMaxScaler()
    normal_data = scaler.fit_transform(dataset)
    normal_data = dataset

    # 画图, 颜色全为coolwarm中的蓝色
    plt.figure(figsize=(4, 3))
    plt.scatter(normal_data[:, 0], normal_data[:, 1], c=label, cmap="coolwarm", alpha=0.5, s=10)
    plt.show()


    # 生成异常点
    # 生成异常点的个数
    # ratio = 0.5   # 散点异常生成比例对应的为：正常数据的size
    # outlier_num = int(ratio * len(normal_data))
    outlier_num = 26
    # 使用均匀分布生成异常点
    # 均匀分布的范围
    x_mean, y_mean = np.mean(normal_data[:, 0]), np.mean(normal_data[:, 1])
    x_min, x_max = np.min(normal_data[:, 0]), np.max(normal_data[:, 0])
    y_min, y_max = np.min(normal_data[:, 1]), np.max(normal_data[:, 1])
    x_range, y_range = max(x_mean - x_min, x_max - x_mean), max(y_mean - y_min, y_max - y_mean)
    # 生成异常点
    outlier_x = np.random.uniform(x_mean - x_range * 1.5, x_mean + x_range * 1.5, outlier_num)
    outlier_y = np.random.uniform(y_mean - y_range * 1.5, y_mean + y_range * 1.5, outlier_num)
    outlier_s = np.vstack((outlier_x, outlier_y)).T

    # 将异常点加入到数据集中
    outlier_data_1 = np.vstack((normal_data, outlier_s))  # 散点异常的数据集
    # 将异常点的标签设置为1, 将原有数据的标签全部设置为0
    label_0 = np.zeros(len(normal_data))
    label_1 = np.ones(len(outlier_s))
    outlier_label_1 = np.hstack((label_0, label_1))

    # 可视化
    plt.figure(figsize=(4, 3))
    plt.scatter(outlier_data_1[:, 0], outlier_data_1[:, 1], c=outlier_label_1, cmap="coolwarm", alpha=0.5, s=10)
    plt.show()

    # 使用高斯分布生成聚集成簇的异常点
    # 生成异常点的个数, 生成比例对应的为：最小簇的size
    # outlier_num = int(ratio * min_cluster_size)
    outlier_num = 26
    # 使用高斯分布生成异常点
    # 定义均值为, 协方差设
    # long1
    # mu = [0, 0.5]
    # cov = [[0.1, 0], [0, 0.0001]]

    #
    # mu2 = [-4, 0.5]
    # cov2 = [[0.01, 0], [0, 0.04]]
    #
    # mu3 = [4, 0.5]
    # cov3 = [[0.006, 0], [0, 0.06]]

    # zelnik3
    mu = [0.45, 0.5]
    cov = [[0.0001, 0], [0, 0.0001]]


    outlier_c1 = np.random.multivariate_normal(mu, cov, outlier_num)
    # outlier_c1中的纵坐标下移 1
    outlier_c1_2 = np.copy(outlier_c1)
    outlier_c1_2[:, 1] = outlier_c1_2[:, 1] - 0.24

    # outlier_num2 = 30
    # outlier_c2 = np.random.multivariate_normal(mu2, cov2, outlier_num2)
    # outlier_num3 = 70
    # outlier_c3 = np.random.multivariate_normal(mu3, cov3, outlier_num3)

    # # 生成异常点
    # outlier2 = np.random.multivariate_normal(mu, cov, outlier_num)

    # 将异常点加入到上面的数据集中
    outlier_data_2 = np.vstack((outlier_data_1, outlier_c1))   # 簇异常 位于簇之间的数据集
    outlier_data_2_2 = np.vstack((outlier_data_1, outlier_c1_2))

    # 将异常点的标签设置为1, 添加到原有标签的后面
    label_2 = np.ones(len(outlier_c1))
    label_2_2 = np.ones(len(outlier_c1_2))
    outlier_label_2 = np.hstack((outlier_label_1, label_2))
    outlier_label_2_2 = np.hstack((outlier_label_1, label_2_2))
    # label 设为整型
    outlier_label_2 = outlier_label_2.astype(int)
    outlier_label_2_2 = outlier_label_2_2.astype(int)

    # 可视化
    plt.figure(figsize=(4, 3))
    plt.scatter(outlier_data_2[:, 0], outlier_data_2[:, 1], c=outlier_label_2, cmap="coolwarm", alpha=0.5, s=10)
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.scatter(outlier_data_2_2[:, 0], outlier_data_2_2[:, 1], c=outlier_label_2_2, cmap="coolwarm", alpha=0.5, s=10)
    plt.show()


    # 保存数据，分别保存
    # 保存数据
    save_path1 = os.path.join(CWD, "data/generate/journal/D11.csv")
    save_data = np.hstack((outlier_data_2, outlier_label_2.reshape(-1, 1)))
    np.savetxt(save_path1, save_data, delimiter=',')

    save_path2 = os.path.join(CWD, "data/generate/journal/D12.csv")
    save_data = np.hstack((outlier_data_2_2, outlier_label_2_2.reshape(-1, 1)))
    np.savetxt(save_path2, save_data, delimiter=',')


    # 检查、可视化保存的数据
    # 读取数据
    data = np.loadtxt(save_path1, delimiter=',')
    # 可视化
    plt.figure(figsize=(4, 3))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.show()

    # 读取数据
    data = np.loadtxt(save_path2, delimiter=',')
    # 可视化
    plt.figure(figsize=(4, 3))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.show()














