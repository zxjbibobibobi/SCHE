import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from utils.utils import fetch_dataset_by_name


def generate_random_numbers(n, random_state, weights=None):
    np.random.seed(random_state)

    # 生成n个随机小数
    if weights is not None:
        random_numbers = np.random.choice(np.arange(1, n + 1), n, p=weights / np.sum(weights), replace=False)
    else:
        random_numbers = np.random.rand(n)
    # 取消随机化
    random_numbers = np.ones(n)
    # 将随机数归一化，使得它们的和为1
    normalized_numbers = random_numbers / np.sum(random_numbers)

    return normalized_numbers


def visualize_binary_2d_dataset(X, y, para):
    colors = ListedColormap(['#1f77b4', '#ff7f0e'])
    imbalance_ratio, mean_loc, covariance_factor = para['IR'], para['mean_loc'], para['overlap']
    # Plot the dataset with improved visualization
    plt.scatter(
        X[:, 0], X[:, 1], c=y, cmap=colors, marker='o', s=10, alpha=0.5, edgecolor='k',
        label=[f'IR:{imbalance_ratio}, LOCATION:{mean_loc}, OVERLAP:{covariance_factor}']
    )
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('CheckBoard Dataset ')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()


def generate_checkboard_dataset(para, if_draw, random_state):
    """该函数接受给定的一组合成数据集制作参数，返回特征向量X和对应的标签y"""

    np.random.seed(random_state)

    imbalance_ratio, mean_loc, covariance_factor = para['IR'], para['mean_loc'], para['overlap']

    center_majority = [
        [0, mean_loc], [0, -1 * mean_loc],
        [-1 * mean_loc, 0],
        [mean_loc, 0],
    ]

    center_minority = [
        [-1 * mean_loc, mean_loc], [-1 * mean_loc, -1 * mean_loc],
        [0, 0],
        [mean_loc, mean_loc], [mean_loc, -1 * mean_loc],
    ]

    cov = [
        [covariance_factor, 0], [0, covariance_factor]
    ]

    cov = np.array(cov)

    n_minority = 1000
    n_majority = n_minority * imbalance_ratio

    # rand_minority_weights = np.array([0.1, 0.1, 0.2, 0.2, 0.4])
    # # rand_minority_weights = None
    # rand_majority_weights = np.array([0.1, 0.2, 0.3, 0.4])

    w_minority = generate_random_numbers(len(center_minority), random_state)
    # print(w_minority)
    w_majority = generate_random_numbers(len(center_majority), random_state)

    # 根据给定的少数类/多数类数量，在各个高斯成分所在的位置生成样本，该比例是随机的
    synthetic_minority, synthetic_majority = np.array([]), np.array([])

    for i in range(len(center_minority)):
        temp = np.random.multivariate_normal(center_minority[i], cov, int(w_minority[i] * n_minority))
        if len(synthetic_minority) == 0:
            synthetic_minority = temp
        else:
            synthetic_minority = np.concatenate((synthetic_minority, temp))

    for i in range(len(center_majority)):
        temp = np.random.multivariate_normal(center_majority[i], cov, int(w_majority[i] * n_majority))
        # print(int(w_minority[i] * n_minority))
        # print(int(w_majority[i] * n_majority))
        if len(synthetic_majority) == 0:
            synthetic_majority = temp
        else:
            synthetic_majority = np.concatenate((synthetic_majority, temp))

    synthetic_dataset = np.concatenate((synthetic_majority, synthetic_minority))
    labels = np.concatenate(
        (np.zeros(len(synthetic_majority)), np.ones(len(synthetic_minority)))
    )

    if if_draw:
        visualize_binary_2d_dataset(synthetic_dataset, labels, para)

    return synthetic_dataset, labels, para

synthetic_para = {
            'IR': 10,
            'mean_loc': 3,
            'overlap': 0.8
        }
generate_checkboard_dataset(synthetic_para, if_draw=True, random_state=42)

colors = ['#74b9ff', '#ff4d4d']

def compress_high_dimension(d_name=None, X=None, y=None, if_draw=True):
    if d_name is not None and X is None:
        X, y = fetch_dataset_by_name(d_name)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    if if_draw:

        markers = ['o', 'o']  # 使用不同的标记
        for label, color, marker in zip(np.unique(y), colors, markers):
            mask = y == label
            alpha = 0.3 if label == 0 else 0.75  # 调整透明度
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], color=color, marker=marker, alpha=alpha, edgecolor='k')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # if d_name is not None:
        # plt.title('Compressed Data Visualization of ' + d_name)
        plt.axis('off')
        plt.show()
    return X_2d, y



def load_imb_toy_dataset(d_name):
    from imblearn.datasets import fetch_datasets
    dataset = fetch_datasets()[d_name]
    X = dataset.data
    y = dataset.target
    y[y == -1] = 0
    return X, y


# for toy in ['optical_digits', 'coil_2000', 'letter_img', 'webpage', 'abalone_19', 'protein_homo']:
# for toy in ['thyroid_sick']:
#     # colors = ListedColormap(['#1f77b4', '#ff7f0e'])
#     X, y = load_imb_toy_dataset(toy)
#     X_2d, y_2d = compress_high_dimension(d_name=toy, X=X, y=y, if_draw=True)
#     X_2d_maj = X_2d[y_2d == 0]
#     X_2d_min = X_2d[y_2d == 1]
#
#     plt.scatter(X_2d_maj[:, 0], X_2d_maj[:, 1], color=colors[0], alpha=0.8, edgecolor='k')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.axis('off')
#     plt.xlim(np.min(X_2d[:, 0]), np.max(X_2d[:, 0]))  # 设置x轴范围
#     plt.ylim(np.min(X_2d[:, 1]), np.max(X_2d[:, 1]))  # 设置y轴范围
#     plt.show()
#
#     plt.scatter(X_2d_min[:, 0], X_2d_min[:, 1], color=colors[1], alpha=0.8, edgecolor='k')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.axis('off')
#     plt.xlim(np.min(X_2d[:, 0]), np.max(X_2d[:, 0]))  # 设置x轴范围
#     plt.ylim(np.min(X_2d[:, 1]), np.max(X_2d[:, 1]))  # 设置y轴范围
#     plt.show()



