from utils import *
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

print('?')


def calculate_derivative(distances):
    return np.gradient(np.gradient(distances))


def multiple_index_selection(data, eps_list, start=None, end=None, step=None):
    sc_list = []
    dbs_list = []
    chs_list = []
    final_index_list = []
    if eps_list is None:
        for eps in np.arange(start, end, step):
            dbscan = DBSCAN(min_samples=5, eps=eps)
            labels = dbscan.fit_predict(data)
            sc_list.append(silhouette_score(data, labels))
            dbs_list.append(davies_bouldin_score(data, labels))
            chs_list.append(calinski_harabasz_score(data, labels))
            final_index_list.append(sc_list[-1] * chs_list[-1] / dbs_list[-1])
    else:
        for eps in eps_list:
            dbscan = DBSCAN(min_samples=5, eps=eps)
            labels = dbscan.fit_predict(data)
            sc_list.append(silhouette_score(data, labels))
            dbs_list.append(davies_bouldin_score(data, labels))
            chs_list.append(calinski_harabasz_score(data, labels))
            final_index_list.append(sc_list[-1] * chs_list[-1] / dbs_list[-1])
    # 选取final_index_list最大的eps
    eps_index = np.argsort(final_index_list)
    max_score = max(final_index_list)
    temp_eps = eps_index[final_index_list==max_score]
    min_eps_index = min(temp_eps)
    selected_eps = eps_list[min_eps_index]
    return selected_eps


# dataset_name = 'payment_simulation'
# RUNS = 10
# RANDOM_STATE = 42
# MAX_INT = np.iinfo(np.int32).max
# seeds = check_random_state(RANDOM_STATE).randint(MAX_INT, size=RUNS)
# test_size = 0.2
# STEP = 0.01
# print('load data')
# X_train, y_train, X_test, y_test = load_dataset_by_name(
#     name=dataset_name, test_size=test_size, randon_state=RANDOM_STATE
# )
# X_min_train = X_train[y_train == 1]
#
# nbr_k = 8
# nbrs = NearestNeighbors(n_neighbors=nbr_k).fit(X_min_train)
#
# # 计算每个点的第k个最近邻距离
# distances, indices = nbrs.kneighbors(X_min_train)
#
# # 对距离进行排序
# distances = np.sort(distances, axis=0)
#
# # 计算距离的增长率
# growth_rate = np.diff(distances[:, nbr_k - 1])
#
# # 寻找距离增长幅度减缓的位置（拐点）
# turning_point_index = np.argmax(growth_rate)
#
# # 绘制K距离图
# plt.figure(figsize=(10, 6))
# plt.plot(distances[:, nbr_k - 1], marker='o', label=f'{nbr_k}-th nearest neighbor distance')
# plt.scatter(turning_point_index, distances[turning_point_index, nbr_k - 1], color='red', label='Turning Point')
# plt.title(f'K-distance Graph with Turning Point (k={nbr_k})')
# plt.xlabel('Points sorted by distance')
# plt.ylabel(f'{nbr_k}-th nearest neighbor distance')
# plt.legend()
# plt.show()
#
# # 输出拐点的位置和距离
# print(f'Turning Point Index: {turning_point_index}')
# print(f'Turning Point Distance: {distances[turning_point_index, nbr_k - 1]}')
#
# percentage2keep = 0.15
# endpoint = 0.9
# brute_list = distances[:, nbr_k - 1][int(len(distances) * (1 - percentage2keep)): int(len(distances) * endpoint)]
# print(brute_list)
#
#
# selected_eps = multiple_index_selection(X_min_train, eps_list=brute_list)
#
# print(selected_eps)