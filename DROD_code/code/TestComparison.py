# pyod中的对比方法。
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 20:04
# @Author  : Sheng
# @File    : TestComparison.py
# @Software: PyCharm
import matplotlib
from matplotlib import pyplot as plt
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import roc_curve, auc
from pyod.utils.data import evaluate_print
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pyod.utils.utility import precision_n_scores
from pyod.models.rod import ROD
from pyod.models.copod import COPOD

matplotlib.use("Qt5Agg")

CWD = os.path.dirname(os.getcwd())  # current working directory to save results to

# filestr = ["Cardiotocography", "fault", "glass", "Ionosphere", "landsat", "letter", "mnist", "musk","optdigits",
#            "pendigits", "PageBlocks", "SpamBase", "speech", "Stamps", "vertebral", "thyroid", "Waveform", "WPBC",
#            "yeast", "Lymphography", "Pima", "satellite", "satimage-2", "vowels", "cardio"]

# filestr = ["Seismic", "Banknote", "Digits", "HeartDisease"]
# filestr = ["data_01", "data_02", "data_03", "data_04", "data_05", "data_06", "data_07", "data_08",
#            "data_09", "data_10", "data_11", "data_12"]
# filestr = ["5_1", "5_2", "10_1", "10_2", "20_1", "20_2", "40_1", "40_2", "60_1", "60_2", "80_1", "80_2"]
# filestr = ["5", "10",  "20",  "30", "40", "50"]
# filestr = ["D9", "D10", "D11", "D12"]

# filestr = ["Cardiotocography", "Ionosphere", "landsat", "optdigits",
#            "pendigits", "PageBlocks","speech",  "Waveform", "WPBC",
#             "Lymphography", "Pima", "satellite", "satimage-2", "vowels", "cardio"]
# filestr = ["Cardiotocography"]
# filestr = ["mnist", "musk","thyroid"]
# filestr = ["Seismic", "Banknote"]
# 生成数据集：D1-D12
# filestr = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12"]
filestr = ["D2"]
# s = 52
for str1 in filestr:

    print("-------------------------",str1,"-------------------------")

    # load dataset
    # data_path = os.path.join(CWD, "data/anomalynpz/{}.npz".format(str1))
    # data = np.load(data_path)
    # dataset = data['X']
    # label = data['y']
    # label = label.reshape(-1, 1)
    # "归一化（0，1）"
    # scaler = MinMaxScaler()
    # normal_data = scaler.fit_transform(dataset)

    # data_path = os.path.join(CWD, "data/anomalycsv/{}.csv".format(str1))
    # df = pd.read_csv(data_path, header= None )  #对于没有列属性的，即第一行为数据 header=None
    # data = df.values
    # dataset = data[:,:-1]
    # label = data[:, -1]
    # # "归一化（0，1）"
    # scaler = MinMaxScaler()
    # normal_data = scaler.fit_transform(dataset)

    """synthetic数据集"""
    # data_path = os.path.join(CWD, "data/syntdata/clusteredscattered/long1_outlier_{}.csv".format(str1))
    # df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
    # data = df.values
    # dataset = data[:, :-1]
    # label = data[:, -1]
    # # "归一化（0，1）"
    # scaler = MinMaxScaler()
    # normal_data = scaler.fit_transform(dataset)

    # data_path = os.path.join(CWD, "data/generate/clustered_scattered/zelnik3_{}.csv".format(str1))
    # data_path = os.path.join(CWD, "data/generate/clustered/{}_cluster1.csv".format("zelnik3"))


    data_path= os.path.join(CWD, "data/generate/journal/{}.csv".format(str1))
    df = pd.read_csv(data_path, header=None)  # 对于没有列属性的，即第一行为数据 header=None
    data = df.values
    dataset = data[:, :-1]
    label = data[:, -1]
    # "归一化（0，1）"
    scaler = MinMaxScaler()
    normal_data = scaler.fit_transform(dataset)

    # IForest 固定seed，取平均值
    AUCIF = []
    PRIF = []
    AUCPRIF = []
    seed_set = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]

    for seed in seed_set:
        iforest = IForest(random_state=seed)
        iforest.fit(normal_data)
        iforest_score = iforest.decision_scores_


        aucIforest = roc_auc_score(label, iforest_score)
        prIforest = precision_n_scores(label, iforest_score)
        precision, recall, _ = precision_recall_curve(label, iforest_score)
        aucprIforest = auc(recall, precision)
        AUCIF.append(aucIforest)
        PRIF.append(prIforest)
        AUCPRIF.append(aucprIforest)

    avg_aucIF = np.mean(AUCIF)
    avg_prIF = np.mean(PRIF)
    avg_aucprIF = np.mean(AUCPRIF)
    print("Dataset: ", str1, " AUC: ", avg_aucIF, " PR: ", avg_prIF, " AUCPR: ", avg_aucprIF)

    # 保存结果到文件
    filepath = os.path.join(CWD, "result/avgIForest.txt")
    with open(filepath, "a") as f:
        f.write("Dataset: " + str1 + " AUC: " + str(avg_aucIF) + " PR: " + str(avg_prIF) + " AUCPR: " + str(avg_aucprIF) + "\n")

    #
    # CBLOF
    cblof = CBLOF(n_clusters=10)
    cblof.fit(normal_data)
    cblof_score = cblof.decision_scores_

    aucCB = roc_auc_score(label, cblof_score)
    prCB = precision_n_scores(label, cblof_score)
    precision, recall, _ = precision_recall_curve(label, cblof_score)
    aucprCB = auc(recall, precision)
    print("Dataset: ", str1, " AUC: ", aucCB, " PR: ", prCB, " AUCPR: ", aucprCB)
    #
    # 写入文件
    filepath = os.path.join(CWD, "result/CBLOF.txt")
    with open(filepath, "a") as f:
        f.write("Dataset: " + str1 + " AUC: " + str(aucCB) + " PR: " + str(prCB) + " AUCPR: " + str(aucprCB) + "\n")

    #
    #ECOD
    ecod = ECOD()
    ecod.fit(normal_data)
    ecod_score = ecod.decision_scores_

    aucECOD = roc_auc_score(label, ecod_score)
    prECOD = precision_n_scores(label, ecod_score)
    precision, recall, _ = precision_recall_curve(label, ecod_score)
    aucprECOD = auc(recall, precision)
    print("Dataset: ", str1, " AUC: ", aucECOD, " PR: ", prECOD, " AUCPR: ", aucprECOD)
    #
    # 写入文件
    filepath = os.path.join(CWD, "result/ECOD.txt")
    with open(filepath, "a") as f:
        f.write("Dataset: " + str1 + " AUC: " + str(aucECOD) + " PR: " + str(prECOD) + " AUCPR: " + str(aucprECOD) + "\n")
    #
    # COPOD
    copod = COPOD()
    copod.fit(normal_data)
    copod_score = copod.decision_scores_

    aucCOPOD = roc_auc_score(label, copod_score)
    prCOPOD = precision_n_scores(label, copod_score)
    precision, recall, _ = precision_recall_curve(label, copod_score)
    aucprCOPOD = auc(recall, precision)
    print("Dataset: ", str1, " AUC: ", aucCOPOD, " PR: ", prCOPOD, " AUCPR: ", aucprCOPOD)
    #
    # # 写入文件
    filepath = os.path.join(CWD, "result/COPOD.txt")
    with open(filepath, "a") as f:
        f.write("Dataset: " + str1 + " AUC: " + str(aucCOPOD) + " PR: " + str(prCOPOD) + " AUCPR: " + str(aucprCOPOD) + "\n")


    # OCSVM
    ocsvm = OCSVM()
    ocsvm.fit(normal_data)
    ocsvm_score = ocsvm.decision_scores_

    aucOCSVM = round(roc_auc_score(label, ocsvm_score),4)
    # 4位小数

    prOCSVM = round(precision_n_scores(label, ocsvm_score),4)
    precision, recall, _ = precision_recall_curve(label, ocsvm_score)
    aucprOCSVM = round(auc(recall, precision),4)

    print("Dataset: ", str1, " AUC: ", aucOCSVM, " PR: ", prOCSVM, " AUCPR: ", aucprOCSVM)
    # # # 写入文件
    filepath = os.path.join(CWD, "result/OCSVM.txt")
    with open(filepath, "a") as f:
        f.write("Dataset: " + str1 + " AUC: " + str(aucOCSVM) + " PR: " + str(prOCSVM) + " AUCPR: " + str(aucprOCSVM) + "\n")







