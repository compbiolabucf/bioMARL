import torch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE
from tqdm import tqdm
from biomarl.feature_env import FeatureEvaluator
from biomarl.utils.logger import info

import numpy as np
import datetime
import argparse
import os
import pandas as pd

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

def gen_rfe(fe: FeatureEvaluator, k=1000):
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    
    if fe.task_type == 'reg':
        estimator = RandomForestRegressor(random_state=0, n_jobs=1)
    else:
        estimator = RandomForestClassifier(random_state=0, n_jobs=1)
        
    selector = RFE(estimator, n_features_to_select=k, step=0.15)
    selector.fit(x, y)
    
    rankings = selector.ranking_
    choice = torch.FloatTensor([1 if rank == 1 else 0 for rank in rankings])
    
    test_result_roc, test_result_pr_auc, original_auc_roc, original_pr_auc = fe.report_performance(choice, flag='test', store=False)
    
    # Get the indices of selected features (where ranking == 1)
    selected_indices = [idx for idx, rank in enumerate(rankings) if rank == 1]
    
    # Sort selected indices by feature importance
    selected_importances = selector.estimator_.feature_importances_[selected_indices]
    ranked_indices = [selected_indices[i] for i in np.argsort(selected_importances)[::-1]]
    
    return original_auc_roc, original_pr_auc, test_result_roc, test_result_pr_auc, choice, ranked_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run RFE feature selection')
    parser.add_argument('--k', type=int, default=100, help='Number of top features to select')
    args = parser.parse_args()

    task_names = ['BRCA_ER', 'BRCA_HER2', 'BRCA_PR', 'BRCA_TN', 'LUAD_cls', 'OV_cls']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = 'results_v3'
    os.makedirs(base_results_dir, exist_ok=True)

    summary_file = os.path.join(base_results_dir, f'rfe_all_results_summary_{timestamp}.txt')

    with open(summary_file, 'w') as summary_f:
        summary_f.write(f"RFE Feature Selection Summary - {datetime.datetime.now()}\n\n")

        for task_name in tqdm(task_names, desc="Processing tasks"):
            print(f"\n{'='*50}")
            print(f'Running RFE feature selection for {task_name}')
            
            file_path = f'data/{task_name}.hdf'
            data = pd.read_hdf(file_path)
            gene_names = data.columns[:-1].tolist()
            
            all_results = []
            
            for run_idx in tqdm(range(1, 11), desc=f"Runs for {task_name}"):
                print(f'Run {run_idx}')
                fe = FeatureEvaluator(task_name)
                original_auc_roc, original_pr_auc, test_roc, test_pr, choice, ranked_indices = gen_rfe(fe, k=args.k)
                
                ranked_genes = [gene_names[i] for i in ranked_indices]
                
                result = {
                    'run': run_idx,
                    'original_roc': original_auc_roc,
                    'original_pr': original_pr_auc, 
                    'test_roc': test_roc,
                    'test_pr': test_pr,
                    'num_selected': int(choice.sum()),
                    'genes': ranked_genes
                }
                all_results.append(result)
                
                print(f"Original ROC AUC: {original_auc_roc:.4f}")
                print(f"Original PR AUC: {original_pr_auc:.4f}")
                print(f"Test ROC AUC: {test_roc:.4f}")
                print(f"Test PR AUC: {test_pr:.4f}")
                print(f"Selected genes: {ranked_genes}")

            # Calculate statistics
            og_rocs = [r['original_roc'] for r in all_results]
            og_prs = [r['original_pr'] for r in all_results]
            test_rocs = [r['test_roc'] for r in all_results]
            test_prs = [r['test_pr'] for r in all_results]

            print(f"\nSummary for {task_name}:")
            print(f"Original ROC: {np.mean(og_rocs):.4f} ± {np.std(og_rocs):.4f}")
            print(f"Original PR: {np.mean(og_prs):.4f} ± {np.std(og_prs):.4f}")
            print(f"Test ROC: {np.mean(test_rocs):.4f} ± {np.std(test_rocs):.4f}")
            print(f"Test PR: {np.mean(test_prs):.4f} ± {np.std(test_prs):.4f}")

            # Write to task-specific file
            task_results_file = os.path.join(base_results_dir, f'rfe_results_{task_name}_{timestamp}.txt')
            with open(task_results_file, 'w') as f:
                f.write(f'RFE results for {task_name}:\n\n')
                for result in all_results:
                    f.write(f"Run {result['run']}:\n")
                    f.write(f"  Original ROC AUC: {result['original_roc']:.4f}\n")
                    f.write(f"  Original PR AUC: {result['original_pr']:.4f}\n")
                    f.write(f"  Test ROC AUC: {result['test_roc']:.4f}\n")
                    f.write(f"  Test PR AUC: {result['test_pr']:.4f}\n")
                    f.write(f"  Selected genes: {result['genes']}\n\n")

                f.write('\nSummary Statistics:\n')
                f.write(f"Original ROC: {np.mean(og_rocs):.4f} ± {np.std(og_rocs):.4f}\n")
                f.write(f"Original PR: {np.mean(og_prs):.4f} ± {np.std(og_prs):.4f}\n")
                f.write(f"Test ROC: {np.mean(test_rocs):.4f} ± {np.std(test_rocs):.4f}\n")
                f.write(f"Test PR: {np.mean(test_prs):.4f} ± {np.std(test_prs):.4f}\n")

                # Add the performance lists section
                f.write('\n\n-- Performance Lists --\n')
                f.write(f'Original ROC Performances: {og_rocs}\n')
                f.write(f'Original PR Performances: {og_prs}\n')
                f.write(f'Test ROC Performances: {test_rocs}\n')
                f.write(f'Test PR Performances: {test_prs}\n')

            # Write summary for this task to the summary file
            with open(summary_file, 'a') as summary_f:
                summary_f.write(f"\nResults for {task_name}:\n")
                summary_f.write(f"Original ROC: {np.mean(og_rocs):.4f} ± {np.std(og_rocs):.4f}\n")
                summary_f.write(f"Original PR: {np.mean(og_prs):.4f} ± {np.std(og_prs):.4f}\n")
                summary_f.write(f"Test ROC: {np.mean(test_rocs):.4f} ± {np.std(test_rocs):.4f}\n")
                summary_f.write(f"Test PR: {np.mean(test_prs):.4f} ± {np.std(test_prs):.4f}\n")

                # Add the performance lists section
                summary_f.write('\n\n-- Performance Lists --\n')
                summary_f.write(f'Original ROC Performances: {og_rocs}\n')
                summary_f.write(f'Original PR Performances: {og_prs}\n')
                summary_f.write(f'Test ROC Performances: {test_rocs}\n')
                summary_f.write(f'Test PR Performances: {test_prs}\n')