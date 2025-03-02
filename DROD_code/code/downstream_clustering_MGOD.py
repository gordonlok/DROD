# 2025.0207
# 用于进行下游聚类任务的 聚类效果实验 mgod的
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 21:43
# @Author  : Sheng
# @File    : downstream_clustering.py
# @Software: PyCharm
import os
import numpy as np
from pyod.utils import evaluate_print
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from MGOD_modify import MGOD_modify
from multiprocessing import Pool


class MULT_dowm:
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

CWD = os.path.dirname(os.getcwd())  # current working directory to save results to

if __name__ == '__main__':
    # load dataset
    data_path = os.path.join(CWD, "data/anomalynpz/{}.npz".format("optdigits"))
    data = np.load(data_path)
    dataset = data['X']
    label = data['y']
    label = label.reshape(-1, 1)
    "归一化（0，1）"
    scaler = MinMaxScaler()
    normal_data = scaler.fit_transform(dataset)
    data = normal_data


    # 使用MGOD
    multd = MULT_dowm(data)
    dwscore = multd.fit_par()

    # auc
    evaluate_print('AUC MGOD:', label, dwscore)
    # # 去除异常分数最高的前150个数据
    as2 = np.array(dwscore)
    index = np.argsort(as2)[::-1][:150]

    data_del_oursample = np.delete(data, index, axis=0)
    label_del_oursample = np.delete(label, index, axis=0)

    # 使用kmeans聚类，k=9
    kmeans_oursample = KMeans(n_clusters=9, random_state=0).fit(data_del_oursample)

    # DBI
    davies_bouldin_oursample = davies_bouldin_score(data_del_oursample, kmeans_oursample.labels_)
    print("davies_bouldin_oursample:", davies_bouldin_oursample)





