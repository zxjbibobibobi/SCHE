

# import packages
import os
import time
import json
import joblib
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
from collections import Counter

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_random_state, check_is_fitted, column_or_1d, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.neighbors import NearestNeighbors

from joblib import Parallel, effective_n_jobs
import functools
from functools import update_wrapper
from contextlib import contextmanager as contextmanager
from sklearn.metrics import recall_score, precision_score, precision_recall_curve, average_precision_score, \
    matthews_corrcoef

import pandas as pd

_global_config = {
    'assume_finite': bool(os.environ.get('SKLEARN_ASSUME_FINITE', False)),
    'working_memory': int(os.environ.get('SKLEARN_WORKING_MEMORY', 1024)),
    'print_changed_only': True,
    'display': 'text',
}

def auc_prc(label, pred):
    return average_precision_score(label, pred)


def hardness_weight_scheduler_function(scheduler_type):
    function_dict = {
        "exp": lambda t, T: np.exp(t * (-1) * np.log(2) / T),
        "linear": lambda t, T: (-1 * t) / (2 * T) + 1,
        "cos": lambda t, T: 0.5 * (1 + np.cos((np.pi * t) / (2 * T))),
        "composite": lambda t, T: 0.75 + 0.5 * np.cos(t * np.pi / T)
    }
    if scheduler_type not in function_dict:
        raise TypeError("function type must be 'exp', 'linear", 'cos', 'composite')

    return function_dict[scheduler_type]


def get_config():
    return _global_config.copy()


def set_config(assume_finite=None, working_memory=None,
               print_changed_only=None, display=None):
    if assume_finite is not None:
        _global_config['assume_finite'] = assume_finite
    if working_memory is not None:
        _global_config['working_memory'] = working_memory
    if print_changed_only is not None:
        _global_config['print_changed_only'] = print_changed_only
    if display is not None:
        _global_config['display'] = display


@contextmanager
def config_context(**new_config):
    old_config = get_config().copy()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)


def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X[:, features])

            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:
                proba[:, estimator.classes_] += \
                    proba_estimator[:, range(len(estimator.classes_))]

        else:
            # Resort to voting
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba



def delayed(function):

    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs

    return delayed_function


class _FuncWrapper:

    def __init__(self, function):
        self.function = function
        self.config = get_config()
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)


def _partition_estimators(n_estimators, n_jobs):
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


class SelfPacedCurriculumEnsemble(BaseEnsemble):

    def __init__(self,
                 estimator=DecisionTreeClassifier(),
                 hardness_func=lambda y_true, y_pred: np.absolute(y_true - y_pred),
                 n_estimators=10,
                 k_bins=10,
                 estimator_params=tuple(),
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 path=None,
                 kdn=None,
                 n_neighbors=None,
                 eps=None,
                 os_flag=False,
                 os_add_flag=False
                 ):

        self.base_estimator = None
        self.hardness_func = hardness_func
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.estimator_params = estimator_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = estimator
        self.path = path
        self.kdn = kdn
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.os_flag = os_flag

    def _random_under_sampling(self, X_maj, y_maj, X_min, y_min):

        np.random.seed(self.random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=False)
        X_train = np.concatenate([X_maj[idx], X_min])
        y_train = np.concatenate([y_maj[idx], y_min])

        return X_train, y_train

    def _self_paced_under_sampling(self, X_maj, y_maj, X_min, y_min, i_estimator):

        hardness = self.hardness_func(y_maj, self.y_maj_pred_proba_buffer[:, self.class_index_min])

        if hardness.max() == hardness.min():
            X_train, y_train = self._random_under_sampling(X_maj, y_maj, X_min, y_min)
        else:
            step = (hardness.max() - hardness.min()) / self.k_bins
            bins = []
            contributions = []
            for i_bins in range(self.k_bins):
                idx = (
                        (hardness >= i_bins * step + hardness.min()) &
                        (hardness < (i_bins + 1) * step + hardness.min())
                )
                if i_bins == (self.k_bins - 1):
                    idx = idx | (hardness == hardness.max())
                bins.append(X_maj[idx])
                contributions.append(hardness[idx].mean())

            alpha = np.tan(np.pi * 0.5 * (i_estimator / (self.n_estimators - 1)))
            weights = 1 / (contributions + alpha)
            weights[np.isnan(weights)] = 0
            n_sample_bins = len(X_min) * weights / weights.sum()
            n_sample_bins = n_sample_bins.astype(int) + 1

            sampled_bins = []
            for i_bins in range(self.k_bins):
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self.random_state)
                    idx = np.random.choice(
                        len(bins[i_bins]),
                        min(len(bins[i_bins]), n_sample_bins[i_bins]),
                        replace=False)
                    sampled_bins.append(bins[i_bins][idx])
            X_train_maj = np.concatenate(sampled_bins, axis=0)
            y_train_maj = np.full(X_train_maj.shape[0], y_maj[0])

            if sp.issparse(X_min):
                X_train = sp.vstack([sp.csr_matrix(X_train_maj), X_min])
            else:
                X_train = np.vstack([X_train_maj, X_min])
            y_train = np.hstack([y_train_maj, y_min])

        return X_train, y_train

    def _validate_y(self, y):

        y = column_or_1d(y, warn=True)
        check_classification_targets(y)

        return y

    def update_maj_pred_buffer(self, X_maj):

        if self.n_buffered_estimators_ > len(self.estimators_):
            raise ValueError(
                'Number of buffered estimators ({}) > total estimators ({}), check usage!'.format(
                    self.n_buffered_estimators_, len(self.estimators_)))
        if self.n_buffered_estimators_ == 0:
            self.y_maj_pred_proba_buffer = np.full(shape=(self._n_samples_maj, self.n_classes_),
                                                   fill_value=1. / self.n_classes_)
        y_maj_pred_proba_buffer = self.y_maj_pred_proba_buffer
        for i in range(self.n_buffered_estimators_, len(self.estimators_)):
        # for i in range(0, len(self.estimators_)):
            y_pred_proba_i = self.estimators_[i].predict_proba(X_maj)
            y_maj_pred_proba_buffer = (y_maj_pred_proba_buffer * i + y_pred_proba_i) / (i + 1)
        self.y_maj_pred_proba_buffer = y_maj_pred_proba_buffer
        self.n_buffered_estimators_ = len(self.estimators_)
        # print(y_maj_pred_proba_buffer[0:5])

        return

    def update_min_pred_buffer(self, X_min, i_iter):
        if self.n_buffered_estimators_ > len(self.estimators_):
            raise ValueError(
                'Number of buffered estimators ({}) > total estimators ({}), check usage!'.format(
                    self.n_buffered_estimators_, len(self.estimators_)))
        if self.n_buffered_estimators_ == 0:
            self.y_min_pred_proba_buffer = np.full(shape=(self._n_samples_min, self.n_classes_),
                                                   fill_value=1. / self.n_classes_)
            # print('once!')
        y_min_pred_proba_buffer = self.y_min_pred_proba_buffer
        for i in range(self.n_buffered_estimators_, len(self.estimators_)):
        # for i in range(0, len(self.estimators_)):
            y_pred_proba_i = self.estimators_[i].predict_proba(X_min)
            y_min_pred_proba_buffer = (y_min_pred_proba_buffer * i + y_pred_proba_i) / (i + 1)
        self.y_min_pred_proba_buffer = y_min_pred_proba_buffer
        self.n_buffered_estimators_ = len(self.estimators_)
        # print(self.n_buffered_estimators_)
        if i_iter == 0:

            return
        else:


            self.record_ih[i_iter] = {}
            self.record_dh[i_iter] = {}


            w_instantaneous_hardness = hardness_weight_scheduler_function("linear")(i_iter, self.n_estimators)
            # w_instantaneous_hardness = 0
            w_local_hardness = 1 - w_instantaneous_hardness
            # w_local_hardness = 0
            instantaneous_hardness = self.hardness_func(self.y_min,
                                                        self.y_min_pred_proba_buffer[:, self.class_index_min])
            os_num_ih = self.os_num_ih
            for label in self.cluster_sample_num:
                self.record_dh[i_iter][str(label)] = {}
                self.record_ih[i_iter][str(label)] = {}
                ih_contribution = []
                n_cluster_sample = self.cluster_sample_num[label]
                for ih in os_num_ih[label]:

                    ih_index = self.X_min_ih_index[label][ih]
                    abs_index = np.array(self.cluster_index[label])[ih_index]
                    ih_instantaneous_hardness = instantaneous_hardness[abs_index]
                    ih_local_hardness = np.full(len(ih_instantaneous_hardness), fill_value=ih)
                    ih_dynamic_hardness = w_instantaneous_hardness * ih_instantaneous_hardness + \
                                          w_local_hardness * ih_local_hardness

                    if ih == 0:
                        pass
                    else:
                        self.record_ih[i_iter][str(label)][ih] = list(ih_dynamic_hardness)
                        self.record_dh[i_iter][str(label)][ih] = list(ih_instantaneous_hardness)
                    ih_contribution.append(sum(ih_dynamic_hardness) / len(ih_dynamic_hardness))
                ih_contribution = np.array(ih_contribution)

                ih_sample_num = n_cluster_sample * ih_contribution / sum(ih_contribution)
                ih_sample_num[np.isnan(ih_sample_num)] = 0
                ih_sample_num[np.isinf(ih_sample_num)] = 0
                ih_sample_num = ih_sample_num.astype(int) + 1
                ih_key = list(os_num_ih[label].keys())
                for i in range(len(ih_sample_num)):
                    os_num_ih[label][ih_key[i]] = ih_sample_num[i]
            self.os_num_ih = os_num_ih
            return

    def init_data_statistics(self, X, y, label_maj, label_min, to_console=False):
        self._n_samples, self.n_features_ = X.shape
        self.features_ = np.arange(self.n_features_)
        self.org_class_distr = Counter(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_buffered_estimators_ = 0

        if self.n_classes_ != 2:
            raise ValueError(f"Number of classes should be 2, meet {self.n_classes_}, please check usage.")

        if label_maj == None or label_min == None:
            # auto detect majority and minority class label
            sorted_class_distr = sorted(self.org_class_distr.items(), key=lambda d: d[1])
            label_min, label_maj = sorted_class_distr[0][0], sorted_class_distr[1][0]
            if to_console:
                print(
                    f'\n\'label_maj\' and \'label_min\' are not specified, automatically set to {label_maj} and {label_min}')

        self.label_maj, self.label_min = label_maj, label_min
        self.class_index_maj, self.class_index_min = list(self.classes_).index(label_maj), list(self.classes_).index(
            label_min)
        maj_index, min_index = (y == label_maj), (y == label_min)
        self._n_samples_maj, self._n_samples_min = maj_index.sum(), min_index.sum()

        if self._n_samples_maj == 0:
            raise RuntimeWarning(
                f'The specified majority class {self.label_maj} has no data samples, please check usage.')
        if self._n_samples_min == 0:
            raise RuntimeWarning(
                f'The specified minority class {self.label_min} has no data samples, please check usage.')

        self.X_maj, self.y_maj = X[maj_index], y[maj_index]
        self.X_min, self.y_min = X[min_index], y[min_index]
        if to_console:
            print('----------------------------------------------------')
            print('# Samples       : {}'.format(self._n_samples))
            print('# Features      : {}'.format(self.n_features_))
            print('# Classes       : {}'.format(self.n_classes_))
            cls_label, cls_dis, IRs = '', '', ''
            min_n_samples = min(self.org_class_distr.values())
            for label, num in sorted(self.org_class_distr.items(), key=lambda d: d[1], reverse=True):
                cls_label += f'{label}/'
                cls_dis += f'{num}/'
                IRs += '{:.2f}/'.format(num / min_n_samples)
            print('Classes         : {}'.format(cls_label[:-1]))
            print('Class Dist      : {}'.format(cls_dis[:-1]))
            print('Imbalance Ratio : {}'.format(IRs[:-1]))
            print('----------------------------------------------------')
            time.sleep(0.25)
        if self.path:
            distances, indices = joblib.load(self.path + str(self.kdn) + 'nbrs_output.joblib')
        else:
            neighbors = NearestNeighbors(n_neighbors=self.kdn, algorithm='ball_tree').fit(X)
            distances, indices = neighbors.kneighbors(self.X_min)
        class1_inst_hardness = []
        class1_indexes = []

        X_maj = X[y == 0]
        X_min = X[y == 1]
        y_maj = y[y == 0]
        y_min = y[y == 1]
        for i in range(len(X_min)):
            # 获取该样本的标签
            label = y_min[i]
            if label == 1:
                nn_indices = indices[i]
                nn_labels = y[nn_indices]

                num_diff_labels = np.sum(nn_labels != label)
                inst_hardness = num_diff_labels / len(nn_indices)
                class1_inst_hardness.append(inst_hardness)
                class1_indexes.append(i)
        class1_inst_hardness = np.array(class1_inst_hardness)
        class1_indexes = np.array(list(range(len(class1_indexes))))

        start = time.time()
        X_min_train_epoch = X_min.copy()
        y_min_train_epoch = y_min.copy()
        noise_num = 0
        dbscan = DBSCAN(eps=self.eps, min_samples=self.n_neighbors, algorithm='auto')
        labels = dbscan.fit_predict(X_min)
        cluster_index = {}
        for index, label in enumerate(labels):
            if label != -1:
                if label not in cluster_index:
                    cluster_index[label] = []
                cluster_index[label].append(index)
            else:
                noise_num += 1
        cluster_size = {}
        max_label = 0
        max_len = 0
        for label in cluster_index.keys():
            cluster_size[label] = len(cluster_index[label])
            if cluster_size[label] > max_len:
                max_len = cluster_size[label]
                max_label = label


        cluster_sample_num = {}
        for label in cluster_index.keys():
            if label != max_label:
                cluster_sample_num[label] = max_len - cluster_size[label]

        X_min_ih_index = {}
        os_num_ih = {}
        for label in cluster_sample_num.keys():
            # 当前簇中样本
            X_sample = X_min[cluster_index[label]]
            cluster_instance_hardness = class1_inst_hardness[cluster_index[label]]

            unique_elements, counts = np.unique(cluster_instance_hardness, return_counts=True)
            ih_contribution = unique_elements * counts

            ih_sample_num = (cluster_sample_num[label] * ih_contribution) / sum(ih_contribution)
            ih_sample_num[np.isnan(ih_sample_num)] = 0
            ih_sample_num[np.isinf(ih_sample_num)] = 0
            ih_sample_num = ih_sample_num.astype(int) + 1
            os_num_ih[label] = {}
            for i in range(len(ih_sample_num)):
                os_num_ih[label][unique_elements[i]] = ih_sample_num[i]


            X_min_ih_index[label] = {}
            for ih in unique_elements:
                X_min_ih_index[label][ih] = (cluster_instance_hardness == ih)

        self.cluster_sample_num = cluster_sample_num
        self.os_num_ih = os_num_ih
        self.cluster_index = cluster_index
        self.X_min_ih_index = X_min_ih_index
        n_total_sample = 0
        for k in cluster_sample_num:
            n_total_sample += cluster_sample_num[k]
        self.n_total_sample = n_total_sample
        return

    def fit(self, X, y, label_maj=None, label_min=None, X_test=None, y_test=None):

        self.record_dh = {}
        self.record_ih = {}

        check_random_state(self.random_state)
        X, y = self._validate_data(
            X, y, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False, multi_output=True)
        y = self._validate_y(y)
        self._validate_estimator()

        self.init_data_statistics(
            X, y, label_maj, label_min,
            to_console=True if self.verbose > 0 else False)
        self.estimators_ = []
        self.estimators_features_ = []

        if self.verbose > 0:
            iterations = tqdm(range(self.n_estimators))
            iterations.set_description('SPE Training')
        else:
            iterations = range(self.n_estimators)
        self.X_min_train_epoch = self.X_min.copy()
        self.y_min_train_epoch = self.y_min.copy()
        self.aucprc = {}

        for i_iter in iterations:


            self.update_maj_pred_buffer(self.X_maj)
            self.update_min_pred_buffer(self.X_min, i_iter)

            if self.os_flag:
                X_min_train_epoch, y_min_train_epoch = self.self_paced_over_sampling_acc()

                self.X_min_train_epoch = X_min_train_epoch
                self.y_min_train_epoch = y_min_train_epoch
            X_train, y_train = self._self_paced_under_sampling(
                self.X_maj, self.y_maj, self.X_min_train_epoch, self.y_min_train_epoch, i_iter)
            estimator = self._make_estimator(append=True, random_state=self.random_state)
            estimator.fit(X_train, y_train)

            if X_test is not None:
                y_pred_proba = np.array(
                    [model.predict_proba(X_test)[:, 1] for model in self.estimators_]
                ).mean(axis=0)
                self.aucprc[i_iter] = auc_prc(y_test, y_pred_proba)
                print(str(i_iter), ':', auc_prc(y_test, y_pred_proba))

            self.estimators_features_.append(self.features_)

        return self

    def _parallel_args(self):
        return {}

    def predict_proba(self, X):

        check_is_fitted(self)
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             **self._parallel_args())(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        proba = sum(all_proba) / self.n_estimators

        return proba

    def predict(self, X):

        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def score(self, X, y):

        return sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, self.class_index_min])

    def self_paced_over_sampling(self):
        X_min_train_epoch = self.X_min_train_epoch.copy()
        y_min_train_epoch = self.y_min_train_epoch.copy()

        for label in self.cluster_sample_num.keys():

            X_sample = self.X_min[self.cluster_index[label]]
            if len(self.os_num_ih) > 0:
                synthetic_samples = IH_SMOTE(cluster_X_min=X_sample, target_index=self.X_min_ih_index[label],
                                             _os_num_ih=self.os_num_ih[label], random_state=self.random_state,
                                             _k=self.n_neighbors, i_iter=self.n_estimators)
                if synthetic_samples is not None:
                    X_min_train_epoch = np.vstack(
                        (X_min_train_epoch, synthetic_samples)
                    )
                    y_min_train_epoch = np.hstack(
                        (y_min_train_epoch, np.ones(len(X_min_train_epoch) - len(y_min_train_epoch))))
            else:
                synthetic_samples = my_SMOTE(_X_min=X_sample, _k=self.n_neighbors, random_state=self.random_state,
                                             _os_num=self.cluster_sample_num[label], i_iter=self.n_estimators)
                if synthetic_samples is not None:
                    X_min_train_epoch = np.vstack(
                        (X_min_train_epoch, synthetic_samples)
                    )
                    y_min_train_epoch = np.hstack(
                        (y_min_train_epoch, np.ones(len(X_min_train_epoch) - len(y_min_train_epoch))))

        return X_min_train_epoch, y_min_train_epoch

    def self_paced_over_sampling_acc(self):
        X_min_train_epoch = self.X_min_train_epoch.copy()
        y_min_train_epoch = self.y_min_train_epoch.copy()

        all_synthetic_samples = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, **self._parallel_args())(
            delayed(_parallel_cluster_sampling)(
                self.X_min, self.cluster_index, label, self.os_num_ih, self.X_min_ih_index, self.random_state,
                self.n_neighbors, self.n_estimators, self.cluster_sample_num
            )for label in self.cluster_sample_num.keys()
        )

        synthetic_samples = []
        for cluster_samples in all_synthetic_samples:
            if len(cluster_samples) == 0:
                pass
            else:
                if len(synthetic_samples) == 0:
                    synthetic_samples = cluster_samples
                else:
                    synthetic_samples = np.vstack((synthetic_samples, cluster_samples))
        if len(synthetic_samples) > 0:
            X_min_train_epoch = np.vstack([synthetic_samples, X_min_train_epoch])
            y_min_train_epoch = np.full(X_min_train_epoch.shape[0], y_min_train_epoch[0])
        return X_min_train_epoch, y_min_train_epoch


def _parallel_cluster_sampling(X_min, cluster_index, label, os_num_ih, X_min_ih_index, random_state, n_neighbors,
                               n_estimators, cluster_sample_num):
    if cluster_sample_num[label] == 0:
        return []
    else:
        X_generated_cluster = []
        X_sample = X_min[cluster_index[label]]
        if len(os_num_ih) > 0:
            synthetic_samples = IH_SMOTE(cluster_X_min=X_sample, target_index=X_min_ih_index[label],
                                         _os_num_ih=os_num_ih[label], random_state=random_state,
                                         _k=n_neighbors, i_iter=n_estimators)
            if synthetic_samples is not None:
                if len(X_generated_cluster) == 0:
                    X_generated_cluster = synthetic_samples
                else:
                    X_generated_cluster = np.vstack(
                        (X_generated_cluster, synthetic_samples)
                    )
        else:
            synthetic_samples = my_SMOTE(_X_min=X_sample, _k=n_neighbors, random_state=random_state,
                                         _os_num=cluster_sample_num[label], i_iter=n_estimators)
            if synthetic_samples is not None:
                if len(X_generated_cluster) == 0:
                    X_generated_cluster = synthetic_samples
                else:
                    X_generated_cluster = np.vstack(
                        (X_generated_cluster, synthetic_samples)
                    )
        return X_generated_cluster
