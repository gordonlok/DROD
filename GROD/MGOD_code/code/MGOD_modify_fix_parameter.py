# 添加多次随机采样, 固定参数
# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 2:22
# @Author  : Sheng
# @File    : MGOD_modify_fix_parameter.py
# @Software: PyCharm


import os

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
from MGOD_modify import MGOD_modify
from multiprocessing import Pool


CWD = os.path.dirname(os.getcwd())  # current working directory to save results to
matplotlib.use("Qt5Agg")

# 多线程
class MULT:

    def __init__(self, data):
        self.data = data
        self.sample_rate = 0.8  # 采样率
        self.sample_times = 60  # 采样次数
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
        seed_set = np.arange(2001, 2061)
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
    # npz真实数据集
    # filestr = ["cardio", "Cardiotocography", "glass", "Ionosphere", "landsat", "Lymphography", "musk", "optdigits",
    #             "PageBlocks", "pendigits", "Pima",  "satellite", "satimage-2", "speech", "Stamps", "thyroid",
    #            "vertebral", "vowels", "Waveform", "WPBC","fault","letter","SpamBase","yeast"]

    # filestr = ["mnist","musk","thyroid"]
    # filestr = ["Cardiotocography",  "glass", "Ionosphere", "musk", "Pima", "Stamps", "thyroid", "WPBC", "Waveform"]
    # filestr = ["musk", "Pima", "Stamps", "thyroid", "WPBC", "Waveform"]
    # filestr = [ "landsat",  "mnist", "optdigits", "Lymphography", "satellite", "satimage-2", "vowels",
    #            "pendigits", "PageBlocks",  "speech", "vertebral"]
    # csv真实数据集
    # filestr = ["Seismic", "Banknote", "HeartDisease", "breastw", "WBC"]
    # filestr = ["Seismic",  "breastw", "WBC"]


    # filestr = ["optdigits"]

    # 选中数据集
    # npz
    # filestr = ["cardio", "Cardiotocography", "glass", "Ionosphere", "landsat", "optdigits",
    #           "PageBlocks", "pendigits", "Pima",  "satellite", "satimage-2", "speech", "Stamps",
    #                     "vowels", "Waveform", "WPBC"]
    # csv
    # filestr = ["Banknote","HeartDisease"]

    # filestr = [ "speech", "Stamps",
    #                     "vowels", "Waveform", "WPBC"]
    #
    # filestr = ["Seismic", "Banknote", "HeartDisease", "breastw", "WBC"]

    # filestr = ["Seismic", "Banknote", "HeartDisease"]


    # 生成数据集
    #filestr = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12"]
    filestr = ["D2"]

    for str1 in filestr:
        print("-------------------------", str1, "-------------------------")
        # load dataset
        # npz真实数据集
        # data_path = os.path.join(CWD, "data/anomalynpz/{}.npz".format(str1))
        # data = np.load(data_path)
        # dataset = data['X']
        # label = data['y']
        # label = label.reshape(-1, 1)
        # "归一化（0，1）"
        # scaler = MinMaxScaler()
        # normal_data = scaler.fit_transform(dataset)
        # data = normal_data

        # csv真实数据集
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
        data_path = os.path.join(CWD, "data/generate/journal/{}.csv".format(str1))
        df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
        data = df.values
        dataset = data[:, :-1]
        label = data[:, -1]
        # "归一化（0，1）"
        scaler = MinMaxScaler()
        normal_data = scaler.fit_transform(dataset)
        data = normal_data


        ourmult = MULT(data)
        anomaly_score = ourmult.fit_par()

        # 计算AUC

        # evaluate_print('Ours:', label, anomaly_score)
        # 保留小数点后四位
        auc1 = round(roc_auc_score(label, anomaly_score), 4)
        print("auc={}".format(auc1))
        # pr@n
        prn = round(precision_n_scores(label, anomaly_score), 4)
        print("prn: ", prn)
        ap = round(average_precision_score(label, anomaly_score), 4)
        print("ap: ", ap)


        # 写入文件
        filepath = os.path.join(CWD, "result/MGOD/fixed_para.txt")
        with open(filepath, "a") as f:
            f.write("-----------------Dataset: " + str1 + "--------------------- \n")
            f.write("AUC: " + str(auc1) + "\n")
            f.write("PR@N: " + str(prn) + "\n")
            f.write("AP: " + str(ap) + "\n")










