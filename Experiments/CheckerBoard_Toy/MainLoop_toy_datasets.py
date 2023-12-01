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
from SCHE.self_paced_curriculum_ensemble import self_paced_curriculum_ensemble
from spe.self_paced_ensemble import self_paced_ensemble
from generate_synthetic_dataset import generate_checkboard_dataset, compress_high_dimension, load_imb_toy_dataset
from experiments.eps_selection import multiple_index_selection
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

"""该脚本测试baseline和SCHE在toy数据集上的表现"""


def knee_visualization(X_min_train, para=None, if_draw=True):
    if para:
        imbalance_ratio, mean_loc, covariance_factor = para['IR'], para['mean_loc'], para['overlap']

    nbr_k = 8
    nbrs = NearestNeighbors(n_neighbors=nbr_k).fit(X_min_train)

    # 计算每个点的第k个最近邻距离
    distances, indices = nbrs.kneighbors(X_min_train)

    # 对距离进行排序
    distances = np.sort(distances, axis=0)

    # 计算距离的增长率
    growth_rate = np.diff(distances[:, nbr_k - 1])

    # 寻找距离增长幅度减缓的位置（拐点）
    turning_point_index = np.argmax(growth_rate)

    if if_draw and para:
        # 绘制K距离图
        plt.figure(figsize=(10, 6))
        plt.plot(distances[:, nbr_k - 1], marker='o',
                 label=[f'IR:{imbalance_ratio}, LOCATION:{mean_loc}, OVERLAP:{covariance_factor}'])
        plt.title(f'K-distance Graph with {nbr_k} nbrs')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{nbr_k}-th nearest neighbor distance')
        plt.legend()
        plt.show()

        # 输出拐点的位置和距离
        print(f'Turning Point Index: {turning_point_index}')
        print(f'Turning Point Distance: {distances[turning_point_index, nbr_k - 1]}')

    percentage2keep = 0.15
    endpoint = 0.9
    brute_list = distances[:, nbr_k - 1][int(len(distances) * (1 - percentage2keep)): int(len(distances) * endpoint)]
    # print(brute_list)

    selected_eps = multiple_index_selection(X_min_train, eps_list=brute_list)

    # print(selected_eps)
    return selected_eps


def main(if_SCHE):
    '''
       开始写循环前，有必要先确定一下baseline。考虑到跑一些smotebagging之类的东西太久了，也不会用到文章实验的第一部分，因此
       以下只使用：SMOTETOMEK,SMOTEENN,Borderline,Tomek,ENN,SPE,DAPS这7种baseline。
       这样选的原因是为了对比混合型采样/混合+集成，同时放一个oversampling方法上去，当然最重要的是对比一下SPE。
       在utils中修改baseline返回代码；
       为了测试数据集有效性，首先对比一下SCHE和SPE的结果，选择较大程度的overlap进行对比
       以下是主循环
    '''
    # if_SCHE = False
    # if_SCHE = True
    if_draw = True

    RUNS = 10
    RANDOM_STATE = 42
    MAX_INT = np.iinfo(np.int32).max
    seeds = check_random_state(RANDOM_STATE).randint(MAX_INT, size=RUNS)
    classifiers = get_classifiers()
    # for toy in ['letter_img', 'webpage', 'optical_digits', 'protein_homo','thyroid_sick']:
    # for toy in ['abalone_19', 'letter_img', 'optical_digits', 'abalone', 'webpage', 'protein_homo']:
    for toy in ['protein_homo']:
    # for toy in ['letter_img']:
    # for toy in ['ecoli', 'optical_digits',
    #             'satimage', 'abalone', 'sick_euthyroid',
    #             'spectrometer', 'isolet',
    #             'us_crime', 'yeast_ml8', 'scene', 'libras_move',
    #             'thyroid_sick', 'coil_2000', 'arrhythmia',
    #             'solar_flare_m0', 'oil', 'car_eval_4',
    #             'wine_quality', 'yeast_me2', 'ozone_level', 'mammography'
    #             ]:
        '''记录一下表现比较好的数据集：
        webpage, abalone_19, protein_homo, letter_img
        pen_digits
        '''
        # try:
        X, y = load_imb_toy_dataset(toy)
        # print(len(X[0]))
        # break
        _, _ = compress_high_dimension(d_name=toy, X=X, y=y, if_draw=if_draw)
        # X, y = generate_toy_dataset()
        X_train, y_train, X_test, y_test = split_imbalance_train_test(X, y, test_size=0.2, random_state=RANDOM_STATE)

        d_name = toy

        if if_SCHE:
            eps = knee_visualization(X_train[y_train == 1], para=None, if_draw=if_draw)
        print("baseline running on dataset: ", d_name)

        # ensemble部分，SPE和DAPS，手动控制SCHE
        for c_index in range(len(classifiers)):
            if not if_SCHE:
                baselines = get_baseline(base_estimator=classifiers[c_index])
            else:
                baselines = [1]
            for b_index in range(len(baselines)):
                # root = './result/toy_SCHE_correct_hardness_same_para_4/' + d_name
                root = './result/toy_SCHE_correct_hardness_same_para_4_only_lh/' + d_name
                results = {
                    'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC',
                                'time']
                }
                results_std = {
                    'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC',
                                'time']
                }
                times, scores = [], []
                if if_SCHE:
                    print('testing baseline: ', 'SelfPacedCurriculumEnsemble')
                    if not os.path.exists(root + '/SCHE_correct_hardness/'):
                        os.makedirs(root + '/SCHE_correct_hardness/')
                    csv_name = root + '/SCHE_correct_hardness/' + classifiers[
                        c_index].__class__.__name__ + '50estimators.csv'
                    std_name = root + '/SCHE_correct_hardness/' + classifiers[
                        c_index].__class__.__name__ + '-std50estimators.csv'
                else:
                    print('testing baseline: ', baselines[b_index].__class__.__name__)
                    if not os.path.exists(root + '/' + baselines[b_index].__class__.__name__ + '/'):
                        os.makedirs(root + '/' + baselines[b_index].__class__.__name__ + '/')
                    csv_name = root + '/' + baselines[b_index].__class__.__name__ + '/' + classifiers[
                        c_index].__class__.__name__ + '50estimators.csv'
                    std_name = root + '/' + baselines[b_index].__class__.__name__ + '/' + classifiers[
                        c_index].__class__.__name__ + '-std50estimators.csv'

                print('running on : ', classifiers[c_index].__class__.__name__)
                print(csv_name)

                try:
                    with trange(RUNS, ncols=80) as t:
                        if if_SCHE:
                            t.set_description_str(
                                'SelfPacedCurriculumEnsemble&' + classifiers[c_index].__class__.__name__ + '&running'
                            )
                        else:
                            t.set_description_str(
                                f'{baselines[b_index].__class__.__name__ + "&" + classifiers[c_index].__class__.__name__}&running')
                        for run_index in t:
                            clf = get_classifiers(random_state=seeds[run_index])[c_index]
                            if if_SCHE:
                                print("SCHE")
                                baseline = self_paced_curriculum_ensemble.SelfPacedCurriculumEnsemble(
                                    estimator=clf, n_estimators=50, kdn=4, eps=eps, n_neighbors=4,
                                    random_state=seeds[run_index],
                                    # path="F:/大四下/毕业设计/project/毕业设计实验代码/nn近邻/",
                                    os_flag=True
                                )
                            else:
                                baseline = get_baseline(base_estimator=clf, random_state=seeds[run_index])[b_index]

                            start = time.time()
                            # baseline.fit(X_train, y_train, X_test=X_test, y_test=y_test)
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
                            print(result)

                except KeyboardInterrupt:
                    t.close()
                print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
                print('------------------------------')
                print('Metrics:')
                df_scores = pd.DataFrame(scores,
                                         columns=['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)',
                                                  'G-mean', 'MCC', 'time'])
                mean_result = []
                std_result = []
                for metric in df_scores.columns.tolist():
                    print(
                        '{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
                    mean_result.append(df_scores[metric].mean())
                    std_result.append(df_scores[metric].std())
                if if_SCHE:
                    results['SelfPacedCurriculumEnsemble'] = mean_result
                    results_std['SelfPacedCurriculumEnsemble'] = std_result
                    print('完成测试', 'SelfPacedCurriculumEnsemble', str(end - start))
                else:
                    results[baselines[b_index].__class__.__name__] = mean_result
                    results_std[baselines[b_index].__class__.__name__] = std_result
                    print('完成测试', baselines[b_index].__class__.__name__, str(end - start))
                # df = pd.DataFrame(results)
                # df_std = pd.DataFrame(results_std)
                # df.to_csv(csv_name, index=False)
                # df_std.to_csv(std_name, index=False)
                # print('保存到本地', csv_name)
        if if_SCHE:
            pass
        else:
            print("开始非集成型算法")
            results = {
                'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time']
            }
            results_std = {
                'metrics': ['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)', 'G-mean', 'MCC', 'time']
            }
            root = './result/' + d_name + '/'
            if not os.path.exists(root):
                os.makedirs(root)
            baselines = get_baseline()
            # non-ensemble 部分，原谅我丑陋的代码, Borderline,ENN,Tomek,SMOTETOMEK,SMTOEENN
            for clf in classifiers:
                results[clf.__class__.__name__] = []
            for b_index in range(len(baselines)):
                print('testing baseline: ', baselines[b_index].__class__.__name__)
                csv_name = root + baselines[
                    b_index].__class__.__name__ + '.csv'
                std_name = root + baselines[
                    b_index].__class__.__name__ + '-std.csv'

                for c_index in range(len(classifiers)):
                    print('using classifier: ', classifiers[c_index].__class__.__name__)
                    times = []
                    scores = []
                    try:
                        with trange(RUNS, ncols=80) as t:
                            t.set_description_str(
                                f'{baselines[b_index].__class__.__name__ + "&" + classifiers[c_index].__class__.__name__}&running')
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
                    except KeyboardInterrupt:
                        t.close()
                    print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
                    print('------------------------------')
                    print('Metrics:')
                    df_scores = pd.DataFrame(scores,
                                             columns=['precision', 'recall', 'f1', 'average Precision-Recall (AUCPRC)',
                                                      'G-mean', 'MCC', 'time'])
                    mean_result = []
                    std_result = []
                    for metric in df_scores.columns.tolist():
                        print('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(),
                                                                   df_scores[metric].std()))
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

                # df.to_csv(csv_name, index=False)
                # df_std.to_csv(std_name, index=False)
                # print('保存到本地', csv_name)
        # except Exception as e:
        #     print(toy, '这个数据集有问题')
        #     print(e)
    return


if __name__ == '__main__':
    # main(False)
    main(True)
