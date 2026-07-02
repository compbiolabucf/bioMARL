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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def gen_rf(fe: FeatureEvaluator, k=1000, random_state=42):
    if fe.task_type == 'reg':
        model = RandomForestRegressor(random_state=random_state)
    elif fe.task_type == 'cls':
        model = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError("Unsupported task type. Only 'reg' and 'cls' are supported.")

    while True:
        x = fe.train.iloc[:, :-1]
        y = fe.train.iloc[:, -1]
        # Convert object columns to numeric or categorical
        for col in x.select_dtypes(include=['object']).columns:
            try:
                x[col] = pd.to_numeric(x[col])
            except ValueError:
                x[col] = x[col].astype('category')

        with tqdm(total=500, desc='Fitting Progress') as pbar:
            model.fit(x, y)
            feature_importances = model.feature_importances_
            pbar.update(400)  # Update progress bar after fitting

        #Normalize importance scores to [0, 1]
        # feature_importances = (feature_importances - feature_importances.min()) / (feature_importances.max() - feature_importances.min())
        # Select top-k features
        top_k_indices = np.argsort(feature_importances)[-k:]
        choice = np.zeros(x.shape[1], dtype=bool)
        choice[top_k_indices] = True
        choice_tensor = torch.FloatTensor(choice)

        test_result_roc, test_result_pr_auc, _, _ = fe.report_performance(choice, flag='test', store=False)
        return test_result_roc, test_result_pr_auc, feature_importances
        
        # test_result, _ = fe.report_performance(choice_tensor, flag='test', store=False)
        # return test_result, feature_importances


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

    print(f'running feature selection for {task_name}')
    fe = FeatureEvaluator(task_name, split=0.3)
    test_roc, test_pr_auc, scores = gen_rf(fe, k = 100)
    print(test_roc, test_pr_auc)
    # print(scores[0:25])
    # print(scores.max(), scores.min())

    print('\n')
    print('**' * 100 )
