# 固定采样率，探索采样时间对MGOD的影响 t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# -*- coding: utf-8 -*-
# @Time    : 2024/6/17 15:57
# @Author  : Sheng
# @File    : MGOD_modify_time.py
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
    # filestr = ["cardio", "Cardiotocography",  "Ionosphere", "landsat", "optdigits",
    #            "PageBlocks", "pendigits", "Pima", "satellite", "satimage-2", "speech",
    #            "vowels", "Waveform", "WPBC"]
    filestr = ["mnist", "musk", "thyroid"]

    # csv
    # filestr = ["Banknote","HeartDisease"]
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
        sample_rate = 0.8
        sample_times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # 手动加上100的结果，减少运行时间
        AUCROC_rate = np.zeros(len(sample_times))
        PR_rate = np.zeros(len(sample_times))
        # average_precision_score
        AP_rate = np.zeros(len(sample_times))



        for r in range(len(sample_times)):
            time = sample_times[r]
            AS_N = np.zeros(N)

            # 多线程
            mult = MULT_sampling(data, sample_rate, time)
            AS_N = mult.fit_par()



            # 保留小数点后四位
            AUCROC_rate[r] = round(roc_auc_score(label, AS_N), 4)
            PR_rate[r] = round(precision_n_scores(label, AS_N), 4)
            AP_rate[r] = round(average_precision_score(label, AS_N), 4)



        # 输出，写入文件
        print('AUCROC:', AUCROC_rate)
        print('PR:', PR_rate)
        print('AP:', AP_rate)

        # 写入文件
        filepath = os.path.join(CWD, "result/MGOD/different_time.txt")
        with open(filepath, "a") as f:
            f.write("-----------------Dataset: " + str1 + "--------------------- \n")
            f.write("AUC: " + "\n")
            f.write(str(AUCROC_rate) + "\n")
            f.write("PR@N: " + "\n")
            f.write(str(PR_rate) + "\n")
            f.write("AP: " + "\n")
            f.write(str(AP_rate) + "\n")

