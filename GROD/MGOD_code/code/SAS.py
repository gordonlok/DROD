# 消融实验，使用子集异常分数SAS作为最终异常分数
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 1:54
# @Author  : Sheng
# @File    : MGOD_modify.py
# @Software: PyCharm


import math
import os
import time
from queue import Queue

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from pyod.utils.utility import precision_n_scores
from pyod.utils.data import evaluate_print
from pyod.models.lof import LOF

class SAS:
    def __init__(self, data):
        self.data = data
        self.row, self.col = data.shape
        self.NaN_density = None


    def find_NaN_auto(self, minK, maxK):
        """
        使用kdtree找出KNN，再找出NaN邻居
        使用多次不增加NAN邻居终止K叠加
        :param K: 下限
        :return: NaN邻居矩阵, NaN距离矩阵
        """
        # 找出每个样本的K个近邻

        nbrs = NearestNeighbors(n_neighbors=maxK, algorithm='ball_tree').fit(self.data)
        knn_dis, knn_indices = nbrs.kneighbors(self.data)


        # 对knn_indices的各行进行检查，若自身索引不是排在第一位，则将自身索引放在第一位，其他索引依次后移
        for i in range(self.row):
            if knn_indices[i, 0] != i:
                # 先检查i是否在自己的KNN中
                if i in knn_indices[i, :]:
                    index = np.argwhere(knn_indices[i, :] == i)
                    index = index[0][0]
                else:
                    continue

                knn_indices[i, 1:index + 1] = knn_indices[i, 0:index]
                knn_indices[i, 0] = i

        # 找出每个样本的NaN邻居
        NaNeighbor_map = np.zeros((self.row, self.row))
        NaN_distance = np.zeros((self.row, self.row))

        # 先进行mink下的找近邻
        K = minK
        for i in range(self.row):
            # i的KNN
            i_knn = knn_indices[i, :K]
            # 索引从1开始，因为0是自身
            for j in range(1, K):
                i_jnn = i_knn[j]
                # j的KNN索引
                j_knn = knn_indices[i_jnn, :K]
                # 如果i在j的KNN中，则i和j是NaN邻居，即互在对方的KNN中
                if i in j_knn:
                    NaNeighbor_map[i, i_jnn] = 1
                    NaNeighbor_map[i_jnn, i] = 1
                    NaN_distance[i, i_jnn] = knn_dis[i, j]  # debug检查距离是否对应
                    NaN_distance[i_jnn, i] = knn_dis[i, j]

        # 统计不存在NaN近邻的点的数量
        zero_num = 0
        for i in range(self.row):
            if np.sum(NaNeighbor_map[i, :]) == 0:
                zero_num += 1
        # print('zero_num: ', zero_num)

        if zero_num == 0:
            # print('一次，所有点都存在NaN近邻')
            return NaNeighbor_map, NaN_distance

        # 对不存在NaN近邻的点，逐渐增加K，直到找到NaN近邻
        Flag = True

        count_old = zero_num
        repeat_time = 0
        next_K = K
        while Flag:
            if K >= maxK:
                # print('K已经达到最大值')
                Flag = False
                break

            next_K = K + 1
            for i in range(self.row):
                # i的第next_K个近邻
                i_k_n = knn_indices[i, next_K - 1]
                # 检查i是否在i_k_n的KNN中
                i_k_n_knn = knn_indices[i_k_n, :next_K]
                if i in i_k_n_knn:
                    NaNeighbor_map[i, i_k_n] = 1
                    NaNeighbor_map[i_k_n, i] = 1
                    NaN_distance[i, i_k_n] = knn_dis[i, next_K - 1]  # debug检查距离是否对应
                    NaN_distance[i_k_n, i] = knn_dis[i, next_K - 1]

            # 统计不存在NaN近邻的点的数量
            zero_num = 0
            for i in range(self.row):
                if np.sum(NaNeighbor_map[i, :]) == 0:
                    zero_num += 1
            # print('zero_num: ', zero_num)

            if zero_num == 0:  # 不存在NaN近邻的点的数量为0
                # print("不存在NaN近邻为0的样本，结束查找")
                Flag = False
                break
            else:
                if zero_num == count_old:
                    repeat_time += 1
                    if repeat_time == 3:
                        # print("重复次数为3，结束查找, zero_num: ", zero_num)
                        # # 输出不存在NaN近邻的点的索引
                        # for i in range(self.row):
                        #     if np.sum(NaNeighbor_map[i, :]) == 0:
                        #         print(i)

                        Flag = False
                        break
                else:
                    repeat_time = 0
                    count_old = zero_num
                    K = next_K

        # print("K: ", next_K)
        return NaNeighbor_map, NaN_distance

    def comp_NaN_density(self, NaNeighbor_map, NaN_distance):
        """
        :param NaNeighbor_map:
        :return: 返回每个点的NaN密度，计算公式为：=  1/(平均距离 + 1/近邻数量)
        """

        NaN_density = np.zeros(self.row)

        for i in range(self.row):
            # 如果i没有NaN邻居，则密度为0
            if np.sum(NaNeighbor_map[i, :]) == 0:
                NaN_density[i] = 0
                continue

            # 局部密度
            NaN_num_i = np.sum(NaNeighbor_map[i, :])
            NaN_num_i = math.log(1+NaN_num_i)
            Sum_NaN_dist = np.sum(NaN_distance[i, :])

            # 平均距离
            avg_NaN_dist = Sum_NaN_dist / NaN_num_i

            NaN_density[i] = 1 / (avg_NaN_dist + 1 / NaN_num_i)

        return NaN_density

    def partition(self, NaNeighbor_map, NaN_distance):
        """

        :param NaNeighbor_map:
        :return:
        """
        nan_dist = NaN_distance
        # 各样本对应标签,初始为-1
        part_map = np.zeros(self.row) - 1
        # 未分配的点，0表示未分配，1表示已分配
        unpart = np.zeros(self.row)

        # 计算根据NaN计算局部密度
        self.NaN_density = self.comp_NaN_density(NaNeighbor_map, nan_dist)

        unpart_density = self.NaN_density.copy()  # NaN_peak_density.copy()   #简单密度下： digits/seismic，可以初定K

        # 初步划分
        flag2 = True
        set = 0
        subset = []
        while (flag2):
            # 在unpart寻找最大密度点
            max_index = np.argmax(unpart_density)
            if unpart_density[max_index] == 0:
                flag2 = False
                break

            if max_index not in subset:
                subset.append(max_index)
            # 是否需要补上max_index 不存在？

            # 定义队列，存储subset中的点
            que = Queue()
            que.put(max_index)
            while (not que.empty()):
                # 取出队列中的第一个点
                q_front = que.get()
                # 找到该点的NaN邻居
                NaNeighbor = np.where(NaNeighbor_map[q_front, :] == 1)
                # 是按最近距离加入subset
                NaN_dis = nan_dist[q_front, NaNeighbor[0]]  # debug检查距离是否对应，不应该存在0值，存在则是bug或者重复
                NaN_dis_min_index = np.argsort(NaN_dis)
                for i in range(len(NaN_dis_min_index)):
                    nan_index = NaNeighbor[0][NaN_dis_min_index[i]]
                    if nan_index not in subset and unpart[nan_index] == 0:
                        subset.append(nan_index)
                        que.put(nan_index)

                # 检查subset大小
                if len(subset) >= np.sqrt(self.row):
                    break

            if len(subset) <= np.sqrt(np.sqrt(self.row)):  # np.sqrt(self.row)/2-1
                for sub1 in subset:
                    part_map[sub1] = -1
                    unpart[sub1] = 1
                    unpart_density[sub1] = 0

                subset = []
                continue

            for sub in subset:
                part_map[sub] = set
                unpart[sub] = 1
                unpart_density[sub] = 0

            # print("set的大小", len(subset))
            set += 1
            subset = []


        # 再划分，将标签为-1的点划分到其他类别
        flag2 = True
        count0 = 0
        while (flag2):
            # 统计标签为-1的点的个数
            count_new = np.sum(part_map == -1)
            if count_new == count0:
                flag2 = False
                break
            count0 = count_new

            for ii in range(self.row):
                if part_map[ii] == -1:
                    # 找到i的邻居
                    NaNeighbor_1 = np.where(NaNeighbor_map[ii, :] == 1)[0]
                    if len(NaNeighbor_1) == 0:
                        continue
                    # 从邻居开始找出邻居的类别

                    # 从最近的邻居开始找
                    NaN_1_dis = nan_dist[ii, NaNeighbor_1]
                    NaN_1_dis_min_index = np.argsort(NaN_1_dis)
                    for jj in range(len(NaN_1_dis_min_index)):
                        nan_1_index = NaNeighbor_1[NaN_1_dis_min_index[jj]]
                        if part_map[nan_1_index] != -1:
                            part_map[ii] = part_map[nan_1_index]
                            break


        return part_map

    def find_set_neighbor(self, part_map, centroid):

        """
        找出每个样本的set neighbor
        :param part_map:
        :return:
        """
        set_num = np.max(part_map) + 1

        set_neighbor = np.zeros(self.row) - 1

        # 计算每个样本的除了自己所在的类别之外最近的其他centriod
        for i in range(self.row):
            # 找到i所在的类别
            i_set = int(part_map[i])

            # 计算第i个数据到其他类别的距离
            dis = np.zeros(set_num)
            for j in range(set_num):
                dis[j] = np.sqrt(np.sum(np.square(self.data[i, :] - centroid[j])))
            # 找到除了自己所在的类别之外最近的其他centriod
            dis[i_set] = 10000  # np.max(dis) #将最大值赋值到自身的距离，这里设为10000，需要修改
            set_neighbor[i] = int(np.argmin(dis))

        return set_neighbor


    def link_strength(self, part_map, centroid, NaN_map, count):

        """
        计算link strength
        :param part_map:
        :param centroid:
        :param NaN_map:
        :param count:
        :return:

        """
        # 子集数, 检测是否等于不同类别数
        set_num = np.max(part_map) + 1

        # 每个样本的除了所属子集簇心最近的簇心
        idata_neighbor = self.find_set_neighbor(part_map, centroid)

        # 只有一个子集时，连接强度为0
        if set_num == 1:
            link_strength = np.zeros((set_num, set_num))
            return link_strength, idata_neighbor

        # link strength
        link_strength = np.zeros((set_num, set_num))

        # 计算每个子集与邻居子集的link strength
        for s in range(set_num):

            # 找到属于s的样本
            s_index = np.where(part_map == s)
            # 在idata_neighbor中找到属于s_index的样本的邻居子集
            s_neighbor = idata_neighbor[s_index]
            # 计算s与s_neighbor的link strength\
            # 去除重复的邻居
            s_neighbor = np.unique(s_neighbor)
            # 子集大小
            s_size = count[s]

            for s_n in range(len(s_neighbor)):


                s_n_index = int(s_neighbor[s_n])
                s_n_size = count[s_n_index]
                if link_strength[s, int(s_neighbor[s_n])] != 0:
                    continue

                # 找出s_neighbor[s_n]的centriod
                s_n_centroid = centroid[int(s_neighbor[s_n]), :]
                # 计算centriod[s]与centriod[s_neighbor[s_n]]的距离
                dis_ci_cj = np.sqrt(np.sum(np.square(s_n_centroid - centroid[s, :])))
                if dis_ci_cj == 0:
                    print("zero")

                # 找出s中的样本与在s_neighbor[s_n]中的样本存在NaN关系的对数。
                nan_in_neighbor = 0
                for i in s_index[0]:
                    # 找出i的NaN邻居
                    i_nan_neighbor = np.where(NaN_map[i, :] == 1)
                    # 找出i_nan_neighbor在s_neighbor[s_n]中的个数
                    for i_i in i_nan_neighbor[0]:
                        if part_map[i_i] == s_neighbor[s_n]:
                            nan_in_neighbor += 1

                # 计算link strength
                ratio = math.log( 1 + nan_in_neighbor)
                #
                link_strength[s, int(s_neighbor[s_n])] = ratio / (dis_ci_cj + 1e-10)
                link_strength[int(s_neighbor[s_n]), s] = link_strength[s, int(s_neighbor[s_n])]


        return link_strength, idata_neighbor

    def calculate_subset_degree(self, link_strength):
        """
        计算子集的度

        :param part_map:
        :param link_strength:
        :return:
        """
        # 统计每个子集与其他子集的link strength
        link_strength_sum = np.sum(link_strength, axis=1)

        # 归一化link_strength_sum，计算每个子集的异常度, ls不用归一化，这里使用归一化或者e
        if np.max(link_strength_sum) != 0:
            link_strength_sum = link_strength_sum / np.max(link_strength_sum)
        else:
            print("连接强度全为0，无法归一化，各子集独立")

        # 计算每个子集的异常度
        subset_degree = 1 - link_strength_sum
        # 检测subset_degree中的元素，若全相等，则赋值全为1
        if np.max(subset_degree) == np.min(subset_degree):
            print("subset_degree 最大 最小值一样，全为1")
            subset_degree = np.ones(len(subset_degree))

        return subset_degree

    def set_max_density_index(self, part_map):
        """
        计算每个子集的最大nan密度点
        :param part_map:
        :param Mean_nan_density:
        :return:
        """
        set_num = np.max(part_map) + 1
        set_max_density_index = np.zeros(set_num)


        for s in range(set_num):
            # 找到属于s的样本
            s_index = np.where(part_map == s)
            # 找出s的最大密度点
            max_index = np.argmax(self.NaN_density[s_index])
            set_max_density_index[s] = s_index[0][max_index]
        return set_max_density_index

    def local_anomaly_score(self, part_map):
        # 有坑点
        # 需要在每个子集中进行计算并归一化，不能直接计算整体的
        # 与子集密度峰值的密度差异作为局部异常分数，局部异常分数越大，越异常
        local_anomaly_score = np.zeros(self.row)
        set_max_density_index = self.set_max_density_index(part_map)

        for i in range(self.row):
            # 找出i的所属子集
            i_set = part_map[i]
            # 标签为-1的，设为最大异常分数
            if i_set == -1:
                continue

            # i的所属子集的最大平均密度的点
            i_set_max_density_index = int(set_max_density_index[i_set])
            i_set_max_density = self.NaN_density[i_set_max_density_index]

            abs_density = np.abs(i_set_max_density - self.NaN_density[i])

            # 绝对密度差异越大，局部异常分数越大
            local_anomaly_score[i] = abs_density

        # 归一化
        if np.max(local_anomaly_score) != np.min(local_anomaly_score):
            local_anomaly_score = (local_anomaly_score - np.min(local_anomaly_score)) / (np.max(local_anomaly_score) - np.min(local_anomaly_score))
        else:
            print("local_anomaly_score can not be normalized")

        return local_anomaly_score


    def fit(self):
        """

        :param
        :return:
        """

        mink = 2  # 1为自身
        maxk = self.row - 1  # 最大为row-1，因为从0开始
        NaN_map, NaN_dist  = self.find_NaN_auto(mink, maxk)  # 未对nan-searching修改

        # 划分子集
        part_map = self.partition(NaN_map, NaN_dist)

        part_map = part_map.astype(np.int32)



        # 统计每个类别的个数 ，-1的未处理
        count = np.zeros(np.max(part_map) + 1)
        for i in range(len(part_map)):
            if part_map[i] != -1:
                count[part_map[i]] += 1
        # print("每个类别个数", count)

        # 将剩余的subset，计算其中心点，这里采用均值
        centroid = []
        for i in range(np.max(part_map) + 1):
            subset = np.where(part_map == i)
            subset_data = self.data[subset[0], :]
            centroid.append(np.mean(subset_data, axis=0))
        centroid = np.array(centroid)

        """subset partition 可视化"""
        # 可视化
        # plt.figure()
        # plt.title("subset partition")
        # plt.scatter(self.data[:, 0], self.data[:, 1], c=part_map, alpha=0.5)
        # for j in range(centroid.shape[0]):
        #     plt.scatter(centroid[j, 0], centroid[j, 1], color='r', marker='*')
        #     plt.text(centroid[j, 0], centroid[j, 1], str(j)+":"+str(count[j]))
        #
        # plt.pause(0.1)


        # 计算子集间link strength
        link_strength, idata_neighbor = self.link_strength(part_map, centroid, NaN_map, count)

        # 计算每个子集的异常度
        subset_degree = self.calculate_subset_degree(link_strength)

        # 获取suset_degree各索引的排名信息，从小到大进行排名。异常程度越高，排名越大，作为e的指数就越大
        subset_degree_rank = subset_degree.argsort().argsort() + 1



        """subset degree 可视化, 值越大，异常度越高，颜色越深"""
        # plt.figure()
        # plt.title("subset degree")
        # # 根据数据对应的part_map，给出每个数据对应子集的异常度，并给出颜色lengend
        # subset_degree_map = np.zeros(self.row)
        # for i in range(self.row):
        #     if part_map[i] == -1:
        #         subset_degree_map[i] = -1
        #     subset_degree_map[i] = subset_degree[part_map[i]]
        # plt.scatter(self.data[:, 0], self.data[:, 1], c=subset_degree_map, alpha=0.5)
        #
        # plt.colorbar()
        # plt.pause(0.1)


        local_anomaly_score = self.local_anomaly_score(part_map)

        # 计算最终异常分数
        score = np.zeros(self.row)
        for i in range(self.row):
            if part_map[i] == -1:
                continue
            else:
                # 计算最终异常分数，消融版SAS 使用subsetdegree作为最终分数
                a = subset_degree[part_map[i]]
                # score[i] = local_anomaly_score[i] * a + subset_degree[part_map[i]]
                score[i] = subset_degree[part_map[i]]

        # 标签为-1的，设为最大异常分数
        for i in range(self.row):
            if part_map[i] == -1:
                score[i] = np.max(score)
        return score


if __name__ == '__main__':

    CWD = os.path.dirname(os.getcwd())  # current working directory to save results to

    filestr = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
    # filestr = ["cardio", "Cardiotocography", "Ionosphere", "landsat", "optdigits",
    #            "PageBlocks", "pendigits", "Pima", "satellite", "satimage-2", "speech",
    #            "vowels", "Waveform", "WPBC"]

    # filestr = ["Banknote", "HeartDisease"]
    for str1 in filestr:
        # # load dataset npz
        # data_path = os.path.join(CWD, "data/anomalynpz/{}.npz".format(str1))
        # data = np.load(data_path)
        # dataset = data['X']
        # label = data['y']
        # label = label.reshape(-1, 1)
        # "归一化（0，1）"
        # scaler = MinMaxScaler()
        # normal_data = scaler.fit_transform(dataset)
        # data = normal_data

        ##  csv
        # data_path = os.path.join(CWD, "data/anomalycsv/{}.csv".format(str1))
        # df = pd.read_csv(data_path, header= None )  #对于没有列属性的，即第一行为数据 header=None
        # data = df.values
        # dataset = data[:,:-1]
        # label = data[:, -1]
        # "归一化（0，1）"
        # scaler = MinMaxScaler()
        # normal_data = scaler.fit_transform(dataset)
        # data = normal_data

        """synthetic数据集"""
        #
        data_path = os.path.join(CWD, "data/generate/D_confer/{}.csv".format(str1))
        df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
        data = df.values
        dataset = data[:, :-1]
        label = data[:, -1]
        # "归一化（0，1）"
        scaler = MinMaxScaler()
        normal_data = scaler.fit_transform(dataset)
        data = normal_data
        data = dataset

        MGOD_SAS = SAS(data)
        score = MGOD_SAS.fit()

        # 保留小数点后四位
        auc = round(roc_auc_score(label, score), 4)
        pn = round(precision_n_scores(label, score), 4)
        ap = round(average_precision_score(label, score), 4)

        print('auc:', auc, '    pr:', pn, "ap: ", ap)

        # 保存实验结果
        save_path = os.path.join(CWD, "result/MGOD/ablation.txt")
        with open(save_path, 'a') as f:
            f.write("Ablation MGOD-SAS: Dataset: " + str(str1) + " AUC: " + str(auc) + " PR: " + str(pn) + "\n")



