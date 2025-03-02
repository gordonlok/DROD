# 同时生成两个inner 和 outer簇异常
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
# filestr = ["long1"]
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
    # plt.title("long1_5%")

    plt.show()


    # 生成簇异常点

    label_0 = np.zeros(len(normal_data))

    outlier_num = 26
    # 使用高斯分布生成异常点
    # 第一个高斯簇
    # 定义均值为[0,0.5], 协方差设
    # zelnik3数据集
    mu = [0.45, 0.5]
    cov = [[0.0001, 0], [0, 0.0001]]

    # long数据集
    # mu = [0, 0.5]
    # cov = [[0.1, 0], [0, 0.0001]]

    outlier_c1 = np.random.multivariate_normal(mu, cov, outlier_num)

    # outlier2 = np.random.multivariate_normal(mu, cov, outlier_num)
    # 将异常点加入到上面的数据集中
    outlier_data_2 = np.vstack((normal_data, outlier_c1))   # 簇异常 位于簇之间的数据集
    # 将异常点的标签设置为1, 添加到原有标签的后面
    label_2 = np.ones(len(outlier_c1))
    outlier_label_2 = np.hstack((label_0, label_2))
    # label 设为整型
    outlier_label_2 = outlier_label_2.astype(int)

    # 可视化
    plt.figure(figsize=(4, 3))
    plt.scatter(outlier_data_2[:, 0], outlier_data_2[:, 1], c=outlier_label_2, cmap="coolwarm", alpha=0.5, s=10)
    plt.show()

    # 使用高斯分布生成第二个数据集
    # 定义均值为[0,0.5], 协方差设
    mu2 = [0.45, 0.2]
    cov2 = [[0.0001, 0], [0, 0.0001]]

    # long数据集
    # mu2 = [0, -0.5]
    # cov2 = [[0.1, 0], [0, 0.0001]]

    outlier_c2 = np.random.multivariate_normal(mu2, cov2, outlier_num)
    # 将异常点加入到上面的数据集中
    outlier_data_3 = np.vstack((outlier_data_2, outlier_c2))  # 簇异常 位于簇外面的数据集
    # 将异常点的标签设置为1, 添加到原有标签的后面
    label_3 = np.ones(len(outlier_c2))
    outlier_label_3 = np.hstack((outlier_label_2, label_3))
    # label 设为单个整数
    outlier_label_3 = outlier_label_3.astype(int)

    # 可视化
    plt.figure(figsize=(4, 3))
    plt.scatter(outlier_data_3[:, 0], outlier_data_3[:, 1], c=outlier_label_3, cmap="coolwarm", alpha=0.5, s=10)
    plt.show()

    # 保存数据

    # save_path1 = os.path.join(CWD, "data/generate/clustered/{}_cluster1.csv".format(str1))

    save_path1 = os.path.join(CWD, "data/generate/journal/D2.csv")
    save_data = np.hstack((outlier_data_3, outlier_label_3.reshape(-1, 1)))
    np.savetxt(save_path1, save_data, delimiter=',')

    # 检查、可视化保存的数据
    # 读取数据
    data = np.loadtxt(save_path1, delimiter=',')
    # 可视化
    plt.figure(figsize=(4, 3))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.show()












