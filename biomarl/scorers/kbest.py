# K-Best Selection.
# The K-Best Selection [3] firstly ranks features by their χ2 scores with the target vector (label vector),
# and then selects the K highest scoring features.
# In the experiments, we make K equal to the number of selected features in MARLFS.
import numpy.random
import torch
import random
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif

from biomarl.feature_env import FeatureEvaluator
from biomarl.utils.logger import info
from collections import Counter

import numpy as np
import datetime
import os
import pandas as pd
import argparse


def gen_kbest(fe: FeatureEvaluator, k = 1000, random_state = 42):
    print(f"\n{'='*50}")
    print(f"Running K-Best selection with k={k}")
    print(f"Original training data shape: {fe.train.shape}")
    # print(f'task type is: {fe.task_type}')
    if fe.task_type == 'reg':
        score_func = f_regression
    elif fe.task_type == 'cls':
        score_func = f_classif
    else:
        score_func = mutual_info_classif

    while True:
        x = fe.train.iloc[:, :-1]
        y = fe.train.iloc[:, -1]
            # Convert object columns to numeric or categorical
        for col in x.select_dtypes(include=['object']).columns:
            try:
                x[col] = pd.to_numeric(x[col])
            except ValueError:
                x[col] = x[col].astype('category')
        # print(x.columns)
        skb = SelectKBest(score_func=score_func, k=k)
        # scores = skb.scores_
        with tqdm(total=500, desc='Fitting Progress') as pbar:
            skb.fit(x, y)
            scores = skb.scores_
            pbar.update(400)  # Update progress bar to 100% after fitting
        # Normalize scores to [0, 1]
        scores = np.nan_to_num(scores, nan=0.0) 
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        choice = torch.FloatTensor(skb.get_support())
        test_result_roc, test_result_pr_auc, _, _ = fe.report_performance(choice, flag='test', store=False)
        return test_result_roc, test_result_pr_auc, scores

if __name__ == '__main__':
    print('=' * 100 )
    print(f"Experimental results from {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
    data_dir = "./data"
    result_dir = "./results"
    # file_names = ["BRCA_ER.hdf", "BRCA_HER2.hdf", "BRCA_PR2.hdf", "BRCA_TN.hdf"]

    parser = argparse.ArgumentParser(description='prefilter dara script')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task (e.g. BRCA_ER)')
    parser.add_argument('--k', type=int, default=1000, help='Number of top features to select')
    args = parser.parse_args()

    file_name = f'{args.task_name}.hdf'
    # file_names = ["LUAD_cls.hdf"]

    task_name = args.task_name
    k = args.k

    print(f'running feature selection for {task_name}')
    fe = FeatureEvaluator(task_name, split=0.3)
    test_result_roc, test_result_pr_auc, scores = gen_kbest(fe, k)
    print(test_result_roc, test_result_pr_auc)
    # print(scores)

    # print(scores.max(), scores.min())


    print('\n')
    print('**' * 100 )

    
