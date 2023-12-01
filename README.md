# Learning from Massive Highly Imbalanced Data via Hybrid-sampling with Self-paced Curriculum  
![SCHE-process](https://github.com/zxjbibobibobi/SCHE/assets/57565621/a7ca65e5-76db-4d50-a3fd-8ad7e809f9fd)

This paper proposes a novel imbalnaced learning framework: Self-paced Curriculum Hybrid-sampling based Ensemble (SCHE), for massive highly imbalanced data classification.

The **Experiments** folder contains the scripts we use for performance evaluation on **CheckerBoard**, **Small-scale Real-world** and **Massive Highly Imbalanced** Datasets.

The implementation details of SCHE is in self_paced_curriculum_ensemble.py.

The reproductivity details of the demonstration in the paper are also contained in our code:3.

Due to limited time, we haven't well organized the whole structure of our project. Soon we will make the code friendly to run.

Wish you a happy day!
## Results on CheckerBoard Datasets
CheckerBoard Datasets
<img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/19bc5280-1d09-45cb-8f90-0390afde7cd2" alt="checkerboard" width="300;">
<img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/e32fbfa6-5976-4d1e-9dd3-0957bfe49f4d" alt="checkerboard" width="300;">
<img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/d2abfac7-fdf1-4327-be8b-bd32bad2a61d" alt="checkerboard_maj" width="300;">

AUPRC variations within varying imbalance ratio (IR) and overlapping level
<img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/6d20138b-7774-4e98-acd9-571f407cdf76" alt="Overlap_AUPRC" width="450;">
<img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/40f6777f-c589-41b5-b30e-7bb8c61ff5dc" alt="IR_AUPRC" width="450;">

AUPRC results on 7 canonical classifiers
<img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/c751869b-4c25-4778-a156-2002379f7cdc" alt="Result on CheckerBoard" width="600;">
