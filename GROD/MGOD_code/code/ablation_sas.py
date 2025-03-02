
import os
import time

import numpy as np
from pyod.utils import evaluate_print, precision_n_scores
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from SAS import SAS
from multiprocessing import Pool
class OurSample:
    def __init__(self, data):
        self.data = data
        self.sample_rate = 0.8
        self.sample_times = 60
        self.N = data.shape[0]

        pass

    def fit(self):

        AS_k = np.zeros(self.N)
        select_times = np.zeros(self.N)
        # 设置种子集，包含60个整数，从2001到2060，方便复现
        seed_set = np.arange(2001, 2061)
        if self.sample_rate == 1:
            sample_data = self.data
            sample_n = sample_data.shape[0]
            sas = SAS(sample_data)
            as2 = sas.fit()
            return as2

        for i in range(self.sample_times):
            # 设置随机种子
            np.random.seed(seed_set[i])
            # sample_idx = np.random.choice(N, int(N * sample_rate), replace=False)
            sample_idx = np.random.choice(self.N, int(self.N * self.sample_rate), replace=False)
            # 在选中的idx，select_times+1
            for id in sample_idx:
                select_times[id] = select_times[id] + 1

            sample_data = self.data[sample_idx]

            sample_n = sample_data.shape[0]

            sample_as = np.zeros(sample_n)


            sas = SAS(sample_data)
            as2 = sas.fit()


            sample_as = as2

            # 将sample_as的值，根据sample_idx的索引，赋值给AS_k
            for id2 in range(sample_n):
                dataidx = sample_idx[id2]
                AS_k[dataidx] = AS_k[dataidx] + sample_as[id2]

            # 进行平均
        AS_k = AS_k / (self.sample_times)

        return AS_k


    def _process_sample(self, seed):
        np.random.seed(seed)
        sample_idx = np.random.choice(self.N, int(self.N * self.sample_rate), replace=False)
        select_times = np.zeros(self.N)
        for id in sample_idx:
            select_times[id] = select_times[id] + 1
        sample_data = self.data[sample_idx]
        sample_n = sample_data.shape[0]
        # Assuming you have defined NaN2 class and its fit method elsewhere
        # nan2 = NaN2(sample_data)
        # as2 = nan2.fit()

        # 消融
        sas = SAS(sample_data)
        as2 = sas.fit()


        return sample_idx, as2

    def fit_par(self):
        seed_set = np.arange(2001, 2061)
        pool = Pool()  # Create a pool of processes
        results = pool.map(self._process_sample, seed_set)  # Map the function to each seed in parallel
        pool.close()
        pool.join()

        AS_k = np.zeros(self.N)

        for sample_idx, sample_as in results:
            for id2 in range(len(sample_idx)):
                dataidx = sample_idx[id2]
                AS_k[dataidx] = AS_k[dataidx] + sample_as[id2]

        AS_k = AS_k / self.sample_times

        return AS_k


if __name__ == '__main__':

    CWD = os.path.dirname(os.getcwd())  # current working directory to save results to

    # filestr = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
    # filestr = ["cardio", "Cardiotocography", "Ionosphere", "landsat", "optdigits",
    #            "PageBlocks", "pendigits", "Pima", "satellite", "satimage-2", "speech",
    #            "vowels", "Waveform", "WPBC"]
    filestr = ["mnist", "musk", "thyroid"]


    # filestr = ["Banknote", "HeartDisease"]
    for str1 in filestr:
        # # load dataset npz
        data_path = os.path.join(CWD, "data/anomalynpz/{}.npz".format(str1))
        data = np.load(data_path)
        dataset = data['X']
        label = data['y']
        label = label.reshape(-1, 1)
        "归一化（0，1）"
        scaler = MinMaxScaler()
        normal_data = scaler.fit_transform(dataset)
        data = normal_data

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
        # data_path = os.path.join(CWD, "data/generate/D_confer/{}.csv".format(str1))
        # df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
        # data = df.values
        # dataset = data[:, :-1]
        # label = data[:, -1]
        # # "归一化（0，1）"
        # scaler = MinMaxScaler()
        # normal_data = scaler.fit_transform(dataset)
        # data = normal_data
        # data = dataset

        # 不采样
        MGOD_SAS = SAS(data)
        score = MGOD_SAS.fit()
        as1 = score


        # 保留小数点后四位
        auc = round(roc_auc_score(label, as1), 4)
        pn = round(precision_n_scores(label, as1), 4)
        ap = round(average_precision_score(label, as1), 4)

        print('auc:', auc, '    pr:', pn,"ap: ", ap)

        # 保存实验结果
        save_path = os.path.join(CWD, "result/MGOD/ablation.txt")
        with open(save_path, 'a') as f:
            f.write("Ablation test: Dataset: " + str(str1) + " AUC: " + str(auc) + " PR: " + str(pn) + "\n")




