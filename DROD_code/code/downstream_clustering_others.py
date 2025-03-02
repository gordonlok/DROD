# 用于进行下游聚类任务的 聚类效果实验 对比方法的
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

CWD = os.path.dirname(os.getcwd())  # current working directory to save results to

import numpy as np


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

# 使用kmeans聚类，k=10

# kmeans_10 = KMeans(n_clusters=10, random_state=0).fit(data)
# kmeans_9 = KMeans(n_clusters=9, random_state=0).fit(data)
#
# # 使用内部指标Davies-Bouldin评估聚类效果

# davies_bouldin_10 = davies_bouldin_score(data, kmeans_10.labels_)
# print("davies_bouldin_10:", davies_bouldin_10)
# davies_bouldin_9 = davies_bouldin_score(data, kmeans_9.labels_)
# print("davies_bouldin_9:", davies_bouldin_9)


# # 随机删除150个数据，使用kmeans聚类，k=9
# index_random = np.random.choice(data.shape[0], 150, replace=False)
# data_del_random = np.delete(data, index_random, axis=0)
# label_del_random = np.delete(label, index_random, axis=0)
#
# kmeans_random_9 = KMeans(n_clusters=9, random_state=0).fit(data_del_random)
#
# print("-----------------------random delete-----------------------")
# # 使用内部指标 Davies-Bouldin评估聚类效果
# davies_bouldin_random_9 = davies_bouldin_score(data_del_random, kmeans_random_9.labels_)
# print("davies_bouldin_random_9:", davies_bouldin_random_9)


# 使用KNN来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.knn import KNN
#
# print("-----------------------KNN-delete-----------------------")
# knn = KNN()
# knn.fit(data)
# as_knn = knn.decision_scores_
#
# # auc
# evaluate_print('AUC KNN:', label, as_knn)


# # 使用IForest来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.iforest import IForest
#
# print("-----------------------IForest-delete-----------------------")
#
# iforest = IForest()
# iforest.fit(data)
# as_iforest = iforest.decision_scores_
#
# # auc
# evaluate_print('AUC IForest:', label, as_iforest)
#

# # 使用LOF来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.lof import LOF
#
# print("-----------------------LOF-delete-----------------------")
#
# lof = LOF()
# lof.fit(data)
# as_lof = lof.decision_scores_
#
# # auc
# evaluate_print('AUC LOF:', label, as_lof)

#
# # 使用CBLOF来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.cblof import CBLOF
#
# print("-----------------------CBLOF-delete-----------------------")
#
# cblof = CBLOF()
# cblof.fit(data)
# as_cblof = cblof.decision_scores_
#
# # auc
# evaluate_print('AUC CBLOF:', label, as_cblof)

#
# # 使用ECOD来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.ecod import ECOD
#
# print("-----------------------ECOD-delete-----------------------")
#
# ecod = ECOD()
# ecod.fit(data)
# as_ecod = ecod.decision_scores_
#
# # auc
# evaluate_print('AUC ECOD:', label, as_ecod)

#
# # 使用COPOD来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.copod import COPOD
#
# print("-----------------------COPOD-delete-----------------------")
#
# copod = COPOD()
# copod.fit(data)
# as_copod = copod.decision_scores_
#

#
# # 使用ABOD来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.abod import ABOD
#
# print("-----------------------ABOD-delete-----------------------")
#
# abod = ABOD()
# abod.fit(data)
# as_abod = abod.decision_scores_
#
# # auc
# evaluate_print('AUC ABOD:', label, as_abod)

# 使用OCSVM来进行异常检测，并去除前150异常分数高的数据
# from pyod.models.ocsvm import OCSVM
#
# print("-----------------------OCSVM-delete-----------------------")
#
# ocsvm = OCSVM()
# ocsvm.fit(data)
# as_ocsvm = ocsvm.decision_scores_
#
# # auc
# evaluate_print('AUC OCSVM:', label, as_ocsvm)


# 使用KNN-based+KFC进行异常检测，并去除前150异常分数高的数据
from KNNKFC import KNNKFC # 需要进去KNNKFC.py文件中修改一下代码
print("-----------------------KNNKFC-delete-----------------------")

knnkfc = KNNKFC()
as_knnkfc = knnkfc.fit(data)

# auc
evaluate_print('AUC KNNKFC:', label, as_knnkfc)

# # 去除异常分数最高的前150个数据
as2 = np.array(as_knnkfc)
index = np.argsort(as2)[::-1][:150]

data_del_oursample = np.delete(data, index, axis=0)
label_del_oursample = np.delete(label, index, axis=0)

# 使用kmeans聚类，k=9
kmeans_oursample = KMeans(n_clusters=9, random_state=0).fit(data_del_oursample)

davies_bouldin_oursample = davies_bouldin_score(data_del_oursample, kmeans_oursample.labels_)
print("davies_bouldin_oursample:", davies_bouldin_oursample)



