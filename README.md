# Learning from Massive Highly Imbalanced Data via Hybrid-sampling with Self-paced Curriculum  
![SCHE-process](https://github.com/zxjbibobibobi/SCHE/assets/57565621/a7ca65e5-76db-4d50-a3fd-8ad7e809f9fd)

This paper proposes a novel imbalnaced learning framework: Self-paced Curriculum Hybrid-sampling based Ensemble (SCHE), for massive highly imbalanced data classification.

The **Experiments** folder contains the scripts we use for performance evaluation on **CheckerBoard**, **Small-scale Real-world** and **Massive Highly Imbalanced** Datasets.

The implementation details of SCHE is in self_paced_curriculum_ensemble.py.

The reproductivity details of the demonstration in the paper are also contained in our code:3.

Due to limited time, we haven't well organized the whole structure of our project. Soon we will make the code friendly to run.

Wish you a happy day! Contact the author at *zxjbibobibobi@163.com*.
## Results on CheckerBoard Datasets
**Visualization of CheckerBoard Dataset**
<center>
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/19bc5280-1d09-45cb-8f90-0390afde7cd2" alt="checkerboard" width="250;">
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/e32fbfa6-5976-4d1e-9dd3-0957bfe49f4d" alt="checkerboard" width="250;">
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/d2abfac7-fdf1-4327-be8b-bd32bad2a61d" alt="checkerboard_maj"width="250;">
</center>

**AUPRC within varying overlapping level (controlled by covariance factor) and imbalance ratio (IR)**
<center>
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/6d20138b-7774-4e98-acd9-571f407cdf76" alt="Overlap_AUPRC" width="400;">
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/40f6777f-c589-41b5-b30e-7bb8c61ff5dc" alt="IR_AUPRC" width="400;">
</center>

**AUPRC results on 7 canonical classifiers**
<center>
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/c751869b-4c25-4778-a156-2002379f7cdc" alt="Result on CheckerBoard" width="900;">
</center>

## Results on Small-scale Real-world Datasets
<center>
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/02ae3cdd-d123-4bcb-8bd8-85ea25d13825" alt="Results on Small-scale Real-world Datasets" width="900;">
</center>

## Results on Massive Highly Imbalanced Datasets
**Compared with 5 baselines**
<center>
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/6c44bb37-ef9f-4390-9512-9445454f62b6" alt="Results on Massive Highly Imbalanced Datasets" width="900;">
</center>

**Compared with resampling methods**
<center>
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/fa77a78a-4d62-449d-a217-a7c0072b701c" alt="Results on Massive Highly Imbalanced Datasets" width="900;">
</center>

**Compared with ensemble methods**
<center>
  <img src="https://github.com/zxjbibobibobi/SCHE/assets/57565621/29de5ffd-96d6-46c9-a49f-426a20e1aee0" alt="Results on Massive Highly Imbalanced Datasets" width="900;">
</center>


