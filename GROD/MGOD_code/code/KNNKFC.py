#
# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 1:42
# @Author  : Sheng
# @File    : KNNKFC.py
# @Software: PyCharm


import os
import pandas as pd

from pyod.utils import precision_n_scores, evaluate_print
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler

CWD = os.path.dirname(os.getcwd())
np.seterr(all="ignore")

class KNNKFC:


    def get_neighbors(self, data, k):
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        return distances, indices.astype(int)

    def cal_outlier_score(self,data, k):
        # 修改异常检测方法
        from pyod.models.lof import LOF
        from pyod.models.knn import KNN
        from pyod.models.abod import ABOD

        clf = KNN(n_neighbors=k)
        clf.fit(data)
        as1 = clf.decision_scores_

        # from DGOF import DGOF
        # clf1 = DGOF(data, k, 0.5)
        # as1 = clf1.fit()


        return as1

    def get_medoid(self,coords):
        coords = np.array(coords)
        cost = distance.cdist(coords, coords, 'cityblock')  # Manhattan distance,  'euclidean','minkowski'
        return np.argmin(cost.sum(axis=0))

    def cal_c_for_KFC(self, outlierScore, data, neighbor, k):
        idx_medoid_object, idx_median_score = np.zeros(data.shape[0]), np.zeros(data.shape[0])
        for i in range(len(data)):
            idx_neighbor = neighbor[i]
            # find index of medoid data
            data_of_neighbors = data[idx_neighbor]
            idx_medoid = self.get_medoid(data_of_neighbors)
            idx_medoid_object[i] = idx_medoid

            # find index of median score
            score_of_neighbors = outlierScore[idx_neighbor]
            idx_median = np.argsort(score_of_neighbors)[len(score_of_neighbors) // 2]
            idx_median_score[i] = idx_median

        u, v = np.array([np.mean(outlierScore[neighbor[i][1:]]) for i in range(len(outlierScore))]), np.array(
            [outlierScore[i] for i in range(len(outlierScore))])
        c_kfcs = 1 - spatial.distance.cosine(u, v)
        u, v = np.array(idx_medoid_object), np.array(idx_median_score)
        c_kfcr = 1 - spatial.distance.cosine(u, v)

        return c_kfcs, c_kfcr

    def FKC(self,k_candidates, data):
        distances, neighbors = self.get_neighbors(data, 100)
        arr_c_kfcs, arr_c_kfcr = [], []
        for k in k_candidates:
            neighbor = neighbors[:, :k + 1]
            outlier_score = self.cal_outlier_score(data, k)
            c_kfcs, c_kfcr = self.cal_c_for_KFC(outlier_score, data, neighbor, k)
            arr_c_kfcs.append(c_kfcs)
            arr_c_kfcr.append(c_kfcr)
        idx_kfcs, idx_kfcr = np.argmax(arr_c_kfcs), np.argmax(arr_c_kfcr)
        return k_candidates[idx_kfcs], k_candidates[idx_kfcr]

    def fit(self,data):# 外部调用时修改，


        k_min, k_max = 3, 99
        k_candidates = range(k_min, k_max)
        optimal_k_by_kfcs, optimal_k_by_kfcr = self.FKC(k_candidates, data)
        print(optimal_k_by_kfcs, optimal_k_by_kfcr)
        a_s = self.cal_outlier_score(data, optimal_k_by_kfcs)

        return a_s





if __name__ == '__main__':

    # DGOF musk k=3出错，thyrpid k=3出错,在mnist上出错，维数太大。
    # npz真实数据集
    # filestr = ["thyroid","mnist","musk"]
    # filestr = ["Cardiotocography",  "glass", "Ionosphere", "musk", "Pima", "Stamps", "thyroid", "WPBC", "Waveform"]
    # filestr = ["mnist"] #DGOF专属,从15-99
    # filestr = [ "optdigits", "Lymphography", "satellite", "satimage-2", "vowels",
    #            "pendigits", "PageBlocks",  "vertebral"]  # 无"speech","mnist" DGOF专属
    # filestr = ["landsat", "mnist", "optdigits", "Lymphography", "satellite", "satimage-2", "vowels",
    #            "pendigits", "PageBlocks", "speech", "vertebral"]

    # filestr = ["Cardiotocography",  "Ionosphere", "landsat", "mnist", "musk","optdigits",
    #            "pendigits", "PageBlocks", "speech",  "thyroid", "Waveform", "WPBC",
    #            "Pima", "satellite", "satimage-2", "vowels", "cardio"]
    # filestr = ["mnist", "musk", "optdigits",
    #            "pendigits", "PageBlocks", "speech", "thyroid", "Waveform", "WPBC",
    #            "Pima", "satellite", "satimage-2", "vowels", "cardio"]

    # filestr = ["PageBlocks","WPBC","Ionosphere", "Waveform","Cardiotocography", "cardio", "landsat",
    #            "pendigits",  "speech",
    #            "Pima", "satellite", "satimage-2", "vowels", ]
    # filestr = [ "cardio"]

    # csv真实数据集
    # filestr = ["Banknote", "HeartDisease"]  # banknote DGOF+KFC的从6开始

    # 生成数据集：D1-D8
    # filestr = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8","D9", "D10", "D11", "D12"]
    # filestr = ["D9", "D10", "D11", "D12"]
    filestr = ["D11", "D12"]
    for str1 in filestr:

        print("-------------------------", str1, "-------------------------")

        # # npz
        # data_path = os.path.join(CWD, "data/anomalynpz/{}.npz".format(str1))
        # data = np.load(data_path)
        # dataset = data['X']
        # label = data['y']
        # label = label.reshape(-1, 1)
        # data = dataset
        # "归一化（0，1）"
        # scaler = MinMaxScaler()
        # normal_data = scaler.fit_transform(dataset)
        # data = normal_data

        # csv
        # data_path = os.path.join(CWD, "data/anomalycsv/{}.csv".format(str1))
        # df = pd.read_csv(data_path, header= None )  #对于没有列属性的，即第一行为数据 header=None
        # data = df.values
        # dataset = data[:,:-1]
        # label = data[:, -1]
        # data = dataset
        # # "归一化（0，1）"
        # scaler = MinMaxScaler()
        # normal_data = scaler.fit_transform(dataset)
        # data = normal_data

        # # D1-D8
        data_path = os.path.join(CWD, "data/generate/journal/{}.csv".format(str1))
        df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
        data = df.values
        dataset = data[:, :-1]
        label = data[:, -1]
        data = dataset

        # "归一化（0，1）"
        scaler = MinMaxScaler()
        normal_data = scaler.fit_transform(dataset)
        data = normal_data



        # find optimal，注意DGOF在数据集musk和throid中k=3会出错,重复值过多建议从5开始。
        k_min, k_max = 3, 99
        k_candidates = range(k_min, k_max)
        kfc = KNNKFC()
        optimal_k_by_kfcs, optimal_k_by_kfcr = kfc.FKC(k_candidates, data)
        print(optimal_k_by_kfcs, optimal_k_by_kfcr)

        # 异常检测方法
        a_s = KNNKFC().cal_outlier_score(data, optimal_k_by_kfcs)
        # 保留小数点后四位
        auc = round(roc_auc_score(label, a_s), 4)
        # 精确度，保留小数点后四位
        pr = round(precision_n_scores(label, a_s), 4)
        ap = round(average_precision_score(label, a_s), 4)
        print('auc:', auc, '    pr:', pr, '    ap:', ap)
        evaluate_print('KFC+KNN_base', label, a_s)

        # as2 = kfc.fit(data)
        # evaluate_print('KFC+KNN_base222', label, as2)

        # 保存实验结果
        save_path = os.path.join(CWD, "result/KNNKFC.txt")
        with open(save_path, 'a') as f:
            f.write("Dataset: " + str1 + " AUC: " + str(auc) + " PR: " + str(pr) + " AP: " + str(ap) + "\n")









