"""
本代码自动测试baseline在4个二分类数据集上的表现，将测试结果自动保存到本地文件夹中
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from tqdm import trange
from utils.utils import *
from imblearn.pipeline import Pipeline
from sklearn.utils import check_random_state
import time
import argparse
import warnings
import pandas as pd


RUNS = 10
RANDOM_STATE = 42
MAX_INT = np.iinfo(np.int32).max
seeds = check_random_state(RANDOM_STATE).randint(MAX_INT, size=RUNS)
test_size = 0.2
# dataset_name = ['yeast4']
# dataset_name = ['yeast4', 'yeast6', 'credit']
dataset_name = ['credit']
# dataset_name = ['payment_simulation']
# dataset_name = ['kddcup99_r2l']
warnings.filterwarnings('ignore')



def parse():
    parser = argparse.ArgumentParser(
        description='Baseline Testing',
        usage='baseline.py  --runs <integer>'
        )
    parser.add_argument('--runs', type=int, default=RUNS, help='Number of independent runs')
    return parser.parse_args()


def test_non_ensemble_baseline():

    root = './result/'
    runs = parse().runs
    for d_name in dataset_name:
        X_train, y_train, X_test, y_test = load_dataset_by_name(
            name=d_name, test_size=test_size, randon_state=RANDOM_STATE
        )
        if not os.path.exists(root + d_name + '/non-ensemble(y9000p)'):
            os.makedirs(root + d_name + '/non-ensemble(y9000p)')
        print("baseline running on dataset: ", d_name)
        results = {
            'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time']
        }
        results_std = {
            'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time']
        }
        classifiers = get_classifiers()
        baselines = get_baseline()
        for clf in classifiers:
            results[clf.__class__.__name__] = []
        for b_index in range(len(baselines)):
            print('testing baseline: ', baselines[b_index].__class__.__name__)
            csv_name = root + d_name + '/non-ensemble(y9000p)' + '/' + baselines[b_index].__class__.__name__ + '.csv'
            std_name = root + d_name + '/non-ensemble(y9000p)' + '/' + baselines[b_index].__class__.__name__ + '-std.csv'

            for c_index in range(len(classifiers)):
                print('using classifier: ', classifiers[c_index].__class__.__name__)
                times = []
                scores = []
                try:
                    with trange(runs, ncols=80) as t:
                        t.set_description_str(f'{baselines[b_index].__class__.__name__+ "&" +classifiers[c_index].__class__.__name__}&running')
                        for run_index in t:
                            baseline = get_baseline(random_state=seeds[run_index])[b_index]
                            clf = get_classifiers(random_state=seeds[run_index])[c_index]
                            model = Pipeline(
                                [
                                    (baseline.__class__.__name__, baseline),
                                    (clf.__class__.__name__, clf)
                                ]
                            )
                            start = time.time()
                            model.fit(X_train, y_train)
                            end = time.time()
                            times.append(end - start)
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            y_pred = model.predict(X_test)
                            result = [
                                precision(y_test, y_pred),
                                recall(y_test, y_pred),
                                f1_optimized(y_test, y_pred_proba),
                                auc_prc(y_test, y_pred_proba),
                                gm_optimized(y_test, y_pred_proba),
                                mcc_optimized(y_test, y_pred_proba),
                                end - start
                            ]
                            scores.append(result)
                            print("第", run_index, "个run")
                            print('precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC',
                                  'time')
                            print(result)
                except KeyboardInterrupt:
                    t.close()
                print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
                print('------------------------------')
                print('Metrics:')
                df_scores = pd.DataFrame(scores, columns=['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time'])
                mean_result = []
                std_result = []
                for metric in df_scores.columns.tolist():
                    print('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
                    mean_result.append(df_scores[metric].mean())
                    std_result.append(df_scores[metric].std())
                results[classifiers[c_index].__class__.__name__] = mean_result
                results_std[classifiers[c_index].__class__.__name__] = std_result
                print('完成测试', classifiers[c_index].__class__.__name__, str(end - start))

            df = pd.DataFrame(results)
            df_std = pd.DataFrame(results_std)
            if os.path.exists(csv_name):
                exist_df = pd.read_csv(csv_name)
                df = pd.concat([df, exist_df], axis=1)

                exist_df_std = pd.read_csv(std_name)
                df_std = pd.concat([df_std, exist_df_std], axis=1)

            df.to_csv(csv_name, index=False)
            df_std.to_csv(std_name, index=False)
            print('保存到本地', csv_name)
    return


def test_ensemble_baseline():
    n_estimators = 50
    print(test_ensemble_baseline)
    classifiers = get_classifiers()
    root = './result/'
    runs = parse().runs

    for d_name in dataset_name:
        X_train, y_train, X_test, y_test = load_dataset_by_name(
            name=d_name, test_size=test_size, randon_state=RANDOM_STATE
        )
        if not os.path.exists(root + d_name + '/ensemble(y9000p)'):
            os.makedirs(root + d_name + '/ensemble(y9000p)')
        print("baseline running on dataset: ", d_name)
        results = {
            'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time']
        }
        results_std = {
            'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time']
        }
        for c_index in range(len(classifiers)):
            baselines = get_baseline(base_estimator=classifiers[c_index])
            csv_name = root + d_name + '/ensemble(y9000p)' + '/' + classifiers[c_index].__class__.__name__ + str(n_estimators) + 'estimators.csv'
            std_name = root + d_name + '/ensemble(y9000p)' + '/' + classifiers[c_index].__class__.__name__ + '-std' + str(n_estimators) + 'estimators.csv'
            for b_index in range(len(baselines)):
                print('testing baseline: ', baselines[b_index].__class__.__name__)
                print('running on : ', classifiers[c_index].__class__.__name__)

                times = []
                scores = []

                try:
                    with trange(runs, ncols=80) as t:
                        t.set_description_str(f'{baselines[b_index].__class__.__name__ + "&" + classifiers[c_index].__class__.__name__}&running')
                        for run_index in t:
                            clf = get_classifiers(random_state=seeds[run_index])[c_index]
                            baseline = get_baseline(n_estimators, clf, random_state=seeds[run_index])[b_index]
                            start = time.time()
                            baseline.fit(X_train, y_train)
                            end = time.time()
                            times.append(end - start)
                            y_pred_proba = baseline.predict_proba(X_test)[:, 1]
                            y_pred = baseline.predict(X_test)
                            result = [
                                precision(y_test, y_pred),
                                recall(y_test, y_pred),
                                f1_optimized(y_test, y_pred_proba),
                                auc_prc(y_test, y_pred_proba),
                                gm_optimized(y_test, y_pred_proba),
                                mcc_optimized(y_test, y_pred_proba),
                                end - start
                            ]
                            scores.append(result)
                            print("第", run_index, "个run")
                            print('precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC',
                                  'time')
                            print(result)
                except KeyboardInterrupt:
                    t.close()

                print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
                print('------------------------------')
                print('Metrics:')
                df_scores = pd.DataFrame(scores, columns=['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time'])
                mean_result = []
                std_result = []
                for metric in df_scores.columns.tolist():
                    print('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
                    mean_result.append(df_scores[metric].mean())
                    std_result.append(df_scores[metric].std())
                results[baselines[b_index].__class__.__name__] = mean_result
                results_std[baselines[b_index].__class__.__name__] = std_result
                print('完成测试', baselines[b_index].__class__.__name__, str(end - start))
            # df = pd.DataFrame(results)
            # df_std = pd.DataFrame(results_std)
            # df.to_csv(csv_name, index=False)
            # df_std.to_csv(std_name, index=False)
            print('保存到本地', csv_name)
    return


def main():
    test_ensemble_baseline()
    # test_non_ensemble_baseline()

if __name__ == '__main__':
    main()
































