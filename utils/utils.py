from sklearn.metrics import recall_score, precision_score, precision_recall_curve, average_precision_score, \
    matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, OneSidedSelection, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier
from self_paced_ensemble.canonical_ensemble import SMOTEBoostClassifier, SMOTEBaggingClassifier, \
    BalanceCascadeClassifier
# from self_paced_ensemble import SelfPacedEnsembleClassifier
# from self_paced_curriculum_ensemble.self_paced_curriculum_ensemble import SelfPacedEnsembleClassifier
import pandas as pd
import numpy as np
from spe.self_paced_ensemble import SelfPacedEnsembleClassifier
from DAPS.DAPS import DAPS

def fetch_dataset_by_name(name):
    """fetching the preprocessed data with the file name"""
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def split_imbalance_train_test(X, y, test_size, random_state=None):
    """ensure the same distribution between split data"""
    X_maj = X[y == 0]
    y_maj = y[y == 0]
    X_min = X[y == 1]
    y_min = y[y == 1]
    X_train_maj, X_test_maj, y_train_maj, y_test_maj = train_test_split(
        X_maj, y_maj, test_size=test_size, random_state=random_state)
    X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(
        X_min, y_min, test_size=test_size, random_state=random_state)
    X_train = np.concatenate([X_train_maj, X_train_min])
    X_test = np.concatenate([X_test_maj, X_test_min])
    y_train = np.concatenate([y_train_maj, y_train_min])
    y_test = np.concatenate([y_test_maj, y_test_min])
    return X_train, y_train, X_test, y_test


def load_dataset_by_name(name, test_size=0.2, randon_state=None):
    X, y = fetch_dataset_by_name(name)
    X_train, y_train, X_test, y_test = split_imbalance_train_test(
        X, y, test_size=test_size, random_state=randon_state
    )
    return X_train, y_train, X_test, y_test


def auc_prc(label, pred):
    return average_precision_score(label, pred)


def precision(label, pred):
    return precision_score(label, pred)


def recall(label, pred):
    return recall_score(label, pred)


def f1_optimized(label, pred):
    pred = pred.copy()
    pre, rec, _ = precision_recall_curve(label, pred)
    return max(2 * pre * rec / (pre + rec))


def gm_optimized(label, pred):
    pred = pred.copy()
    pre, rec, _ = precision_recall_curve(label, pred)
    return max(np.power((pre * rec), 0.5))


def mcc_optimized(label, pred):
    mcc = []
    for t in range(100):
        _pred = pred.copy()
        _pred[_pred < 0 + t * 0.01] = 0
        _pred[_pred >= 0 + t * 0.01] = 1
        mcc.append(matthews_corrcoef(label, _pred))
    return max(mcc)


def get_baseline(n_estimators=100, base_estimator=None, random_state=None):
    if base_estimator is None:
        baseline = [
            # SMOTE(random_state=random_state),
            # ADASYN(random_state=random_state),
            # RandomOverSampler(random_state=random_state),
            # BorderlineSMOTE(random_state=random_state),
            # RandomUnderSampler(random_state=random_state),
            # EditedNearestNeighbours(),
            # OneSidedSelection(random_state=random_state),
            # TomekLinks(),
            # SMOTETomek(random_state=random_state),
            # SMOTEENN(random_state=random_state)
        ]
    else:
        if base_estimator.__class__.__name__ == 'KNeighborsClassifier':
            baseline = [
                SelfPacedEnsembleClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # BalanceCascadeClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # EasyEnsembleClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # SMOTEBaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # BalancedBaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
            ]
        else:
            baseline = [
                SelfPacedEnsembleClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # DAPS(base_estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # BalanceCascadeClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # EasyEnsembleClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # SMOTEBaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # RUSBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # BalancedBaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state),
                # SMOTEBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=random_state)


            ]

    return baseline


def get_classifiers(random_state=None):
    return [
        # AdaBoostClassifier(n_estimators=10),
        DecisionTreeClassifier(random_state=random_state),
        # LogisticRegression(random_state=random_state),
        # KNeighborsClassifier(),
        # RandomForestClassifier(n_estimators=10, random_state=random_state),
        # GradientBoostingClassifier(n_estimators=10, random_state=random_state),
        # SVC(C=1000, probability=True)
    ]
