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
from typing import List, Tuple, Union




from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def gen_svm(fe: FeatureEvaluator, k=1000, kernel='linear'):
    if fe.task_type == 'reg':
        model = make_pipeline(StandardScaler(), SVR(kernel=kernel))
    elif fe.task_type == 'cls':
        model = make_pipeline(StandardScaler(), SVC(kernel=kernel))
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
            pbar.update(400)  # Update progress bar after fitting

        # Calculate feature importance
        if kernel == 'linear':
            importance = np.abs(model[-1].coef_).flatten()
        else:
            # Approximate importance for non-linear kernels via decision function
            support_vectors = model[-1].support_vectors_
            importance = np.sum(np.abs(support_vectors), axis=0)

        # Normalize importance scores to [0, 1]
        # importance = (importance - importance.min()) / (importance.max() - importance.min())
        
        # Select top-k features
        top_k_indices = np.argsort(importance)[-k:]
        choice = np.zeros(x.shape[1], dtype=bool)
        choice[top_k_indices] = True
        choice_tensor = torch.FloatTensor(choice)

        test_result_roc, test_result_pr_auc, _, _ = fe.report_performance(choice, flag='test', store=False)
        return test_result_roc, test_result_pr_auc, importance
        
        # test_result, _ = fe.report_performance(choice_tensor, flag='test', store=False)
        # return test_result, importance


def filter_dataset_by_features(dataset: pd.DataFrame, selected_indices: List[int], gene_names: List[str]) -> pd.DataFrame:
    selected_columns = dataset.columns[selected_indices].tolist() + [dataset.columns[-1]]
    filtered_dataset = dataset[selected_columns]

    #rename columns with gene names
    selected_gene_names = [gene_names[i] for i in selected_indices] + ['target']
    filtered_dataset.columns = selected_gene_names
    return filtered_dataset

def save_filtered_dataset(filtered_dataset: pd.DataFrame, save_path: str, key: str='data'):
    filtered_dataset.to_hdf(save_path, key=key, mode='w')

if __name__ == '__main__':
    print('=' * 100 )
    print(f"Experimental results from {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
    data_dir = "./data"
    result_dir = "./data"
    file_names = ["LUAD_cls.hdf"]

    parser = argparse.ArgumentParser(description='prefilter dara script')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task (e.g. BRCA_ER)')
    parser.add_argument('--k', type=int, default=500, help='Number of top features to select')
    args = parser.parse_args()

    file_name = f'{args.task_name}.hdf'
    # file_names = ["LUAD_cls.hdf"]

    task_name = args.task_name
    k = args.k

    file_path = os.path.join(data_dir, file_name)
    data = pd.read_hdf(file_path)
    gene_names = data.columns[:-1].tolist()
    print(f'running meta feature selection for {task_name}')

    print(f'running feature selection for {task_name}')
    fe = FeatureEvaluator(task_name, split=0.35)
    test_roc_auc, test_pr_auc, scores = gen_svm(fe, k=k, kernel='linear')
    print(test_roc_auc, test_pr_auc)
    # print(scores)

    # Filter dataset by selected features
    selected_indices = np.argsort(scores)[-k:]

    dataset = pd.read_hdf(os.path.join(data_dir, file_name))


    filtered_dataset = filter_dataset_by_features(dataset, selected_indices, gene_names)

    # Save the filtered dataset
    save_path = os.path.join(result_dir, f"prefiltered_svm_{task_name}.hdf")
    save_filtered_dataset(filtered_dataset, save_path)

    print(f"Filtered dataset saved to {save_path}")

    print('\n')
    print('**' * 100 )
