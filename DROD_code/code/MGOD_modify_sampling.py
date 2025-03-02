# 不同采样率实验, 采样次数固定在100次
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 2:25
# @Author  : Sheng
# @File    : MGOD_modify_sampling.py
# @Software: PyCharm

import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

from multiprocessing import Pool
from MGOD_modify import MGOD_modify


class MULT_sampling:
    def __init__(self, data , sample_rate, sample_times):
        self.data = data
        self.sample_rate = sample_rate  # 采样率
        self.sample_times = sample_times  # 采样次数
        self.N = data.shape[0]
        pass


    def _process_sample(self, seed):
        np.random.seed(seed)
        sample_idx = np.random.choice(self.N, int(self.N * self.sample_rate), replace=False)
        select_times = np.zeros(self.N)
        for id in sample_idx:
            select_times[id] = select_times[id] + 1
        sample_data = self.data[sample_idx]
        # sample_n = sample_data.shape[0]

        mgod = MGOD_modify(sample_data)
        score = mgod.fit()

        # nan2
        # nan2 = NaN2(sample_data)
        # score = nan2.fit()

        return sample_idx, score

    def fit_par(self):
        seed_set = np.arange(2001, 2001 + self.sample_times)
        pool = Pool()  # Create a pool of processes
        results = pool.map(self._process_sample, seed_set)  # Map the function to each seed in parallel
        pool.close()
        pool.join()

        AS_N = np.zeros(self.N)
        # select_times = np.zeros(self.N)

        for sample_idx, sample_as in results:
            for id2 in range(len(sample_idx)):
                dataidx = sample_idx[id2]
                AS_N[dataidx] = AS_N[dataidx] + sample_as[id2]

        AS_N = AS_N / self.sample_times

        return AS_N



if __name__ == '__main__':
    CWD = os.path.dirname(os.getcwd())  # current working directory to save results to
    matplotlib.use("Qt5Agg")
    # 选中数据集
    # npz
    # filestr = ["cardio", "Cardiotocography", "glass", "Ionosphere", "landsat", "optdigits",
    #            "PageBlocks", "pendigits", "Pima", "satellite", "satimage-2", "speech", "Stamps",
    #            "vowels", "Waveform", "WPBC"]
    # filestr = [ "pendigits", "Pima", "satellite", "satimage-2", "speech", "Stamps",
    #            "vowels", "Waveform", "WPBC"]

    # csv
    # filestr = ["Banknote","HeartDisease"]
    filestr = ["mnist", "musk", "thyroid"]
    # filestr = ["Seismic", "Banknote", "HeartDisease", "breastw", "WBC"]

    # filestr = ["Pima"]

    for str1 in filestr:

        print("-------------------------", str1, "-------------------------")
        # load dataset
        data_path = os.path.join(CWD, "data/anomalynpz/{}.npz".format(str1))
        data = np.load(data_path)
        dataset = data['X']
        label = data['y']
        label = label.reshape(-1, 1)
        "归一化（0，1）"
        scaler = MinMaxScaler()
        normal_data = scaler.fit_transform(dataset)
        data = normal_data


        # data_path = os.path.join(CWD, "data/anomalycsv/{}.csv".format(str1))
        # df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
        # data = df.values
        # dataset = data[:, :-1]
        # label = data[:, -1]
        # "归一化（0，1）"
        # scaler = MinMaxScaler()
        # normal_data = scaler.fit_transform(dataset)
        # data = normal_data

        N = data.shape[0]

        """定于抽样率、抽样次数"""
        sample_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        sample_times = 100
        AUC_rate = np.zeros(len(sample_rate))
        PR_rate = np.zeros(len(sample_rate))

        for r in range(len(sample_rate)):
            rate = sample_rate[r]
            AS_N = np.zeros(N)

            # 进行异常评分
            if rate == 1:
                # 不需要多线程
                mgod = MGOD_modify(data)
                score = mgod.fit()
                AS_N = score
            else:
                # 多线程
                mult = MULT_sampling(data, rate, sample_times)
                AS_N = mult.fit_par()

            # 保留小数点后四位
            AUC_rate[r] = round(roc_auc_score(label, AS_N), 4)
            PR_rate[r] = round(precision_n_scores(label, AS_N), 4)


        # 输出，写入文件
        print('AUC:', AUC_rate)
        print('PR:', PR_rate)

        # 写入文件
        filepath = os.path.join(CWD, "result/MGOD/different_rate.txt")
        with open(filepath, "a") as f:
            f.write("-----------------Dataset: " + str1 + "--------------------- \n")
            f.write("AUC: " + "\n")
            f.write(str(AUC_rate) + "\n")
            f.write("PR@N: " + "\n")
            f.write(str(PR_rate) + "\n")

















