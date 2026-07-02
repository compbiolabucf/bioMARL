# import torch
# from lassonet import LassoNetClassifierCV, LassoNetRegressorCV
# from sklearn import preprocessing
# from sklearn.preprocessing import LabelEncoder

# from biomarl.feature_env import FeatureEvaluator
# from biomarl.utils.logger import info
# import random

# import numpy as np
# import datetime
# import os
# import pandas as pd


# def gen_lassonet(fe: FeatureEvaluator, limit_1 = 0.15, limit_2 = 0.2):
#     double_limit = True
#     while True:
#         x = fe.train.iloc[:, :-1].to_numpy()
#         y = fe.train.iloc[:, -1].to_numpy()
#         x_columns = x.shape[1]  # Get the number of columns in x
#         # print(x.columns)
        
#         x = x.astype(float)
#         # Handle y based on task type
#         if fe.task_type == 'cls':
#             le = LabelEncoder()
#             y = le.fit_transform(y)
#         k = max(2, int(random.uniform(limit_1 * x_columns, limit_2 * x_columns)))

#         if fe.task_type == 'reg':
#             selector = LassoNetRegressorCV()
#         else:
#             # normalizer = preprocessing.Normalizer()
#             # normalizer.fit(x)
#             # x = normalizer.transform(x)
#             selector = LassoNetClassifierCV()  # LassoNetRegressorCV

#         selector = selector.fit(x, y)
#         scores = selector.feature_importances_
#         value, indice = torch.topk(scores, k)
#         choice = torch.zeros(x.shape[1])
#         choice[indice] = 1
#         if fe.task_type == 'cls':
#             test_result, original_result = fe.report_performance(choice, flag='test', store=False)
#             if (test_result - original_result) < -0.05 * original_result:
#                 if not double_limit:
#                     info(f'Performance improvement {test_result - original_result} is less than -5%, doubling limits and restarting the process.')
#                     limit_1 *= 2
#                     limit_2 *= 1.5
#                     double_limit = True
#                     continue
#                 else:
#                     info('Limits already doubled once, exiting the loop with current results.')
#                     break
#             else:
#                 break
#         else:
#             test_result = fe.report_performance(choice, flag='test', store=False)

#     selected_features_indices = torch.nonzero(choice).squeeze().tolist()
#     ranked_indices = [x for _, x in sorted(zip(scores[selected_features_indices], selected_features_indices), reverse=True)]
#     ranked_features = [fe.train.columns[i] for i in ranked_indices]
#     # result = fe.report_performance(choice, flag='train')
#     # test_result = fe.report_performance(choice, flag='test', store=False)
#     info("The optimal performance is: {}".format(test_result))
#     print(f'optimal number of features to be selected is: {int(choice.sum())}')
#     print(f'ranked features: {ranked_features}')
#     return original_result, test_result, choice, ranked_features

# if __name__ == '__main__':
#     print('=' * 100 )
#     print(f"Experimental results from {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
#     data_dir = "data"
#     result_dir = "results"
#     file_names = ["BRCA_TN.hdf"]
#     # file_names = ['LUAD_cls.hdf', 'OV_cls.hdf']
#     # evals = ['RF', 'XGB', 'DT', 'SVM', 'LASSO', 'Ridge']
#     evals = ['KNN']

#     for eval in evals:
#         for file_name in file_names:
#             task_name = file_name.split('.')[0]
#             file_path = os.path.join(data_dir, file_name)
#             data = pd.read_hdf(file_path)
#             gene_names = data.columns[:-1]

#             all_ranked_genes = []
#             roc_list = [] #keep track fo performance
#             og_perf_list = []

#             best_max_roc = -1
#             best_genes = [] #for the best selection
#             best_num_genes = 0 #number of genes in that selection

#             for run_idx in range(1, 11):
#                 fe = FeatureEvaluator(task_name, method=eval)
#                 print(f'running feature selection for {task_name} with {eval}, run {run_idx}')
#                 og_result, test_result, _, ranked_indices = gen_lassonet(fe, limit_1 = 0.006, limit_2 = 0.012)

#                 ranked_genes = [gene_names[i] for i in ranked_indices]
#                 # print(ranked_genes)tes

#                 all_ranked_genes.append(ranked_genes)
#                 roc_list.append(test_result)
#                 og_perf_list.append(og_result)

#                 if test_result > best_max_roc:
#                     best_max_roc = test_result
#                     best_genes = ranked_genes
#                     best_num_genes = len(ranked_genes)

#             # Calculate and print overall performance metrics
#             overall_mean_accuracy = np.mean(roc_list)
#             overall_variance_accuracy = np.var(roc_list)

#             info(f'All original results: {og_perf_list}')
#             info(f'mean of all og results: {np.mean(og_perf_list)}')
#             info(f'variance of all og performance: {np.var(og_perf_list)}')

#             print('\n')
#             info(f'All results: {roc_list}')
#             info(f"Overall Mean Accuracy for {eval} method is {overall_mean_accuracy}")
#             info(f"Overall Variance of Accuracy: {overall_variance_accuracy}")
#             info(f"Best Max Accuracy: {best_max_roc}")

#             result_file = os.path.join(result_dir, f'final_selection_lassonet_{task_name}.txt')

#             with open(result_file, 'a') as f:
#                 f.write(f"Experimental results from {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
#                 for run_idx, (ranked_genes, roc) in enumerate(zip(all_ranked_genes, roc_list), 1):
#                     f.write(f"Run {run_idx}:\n")
#                     f.write(f"  Genes: {', '.join(ranked_genes)}\n")
#                     f.write(f"      Number of selected genes: {len(ranked_genes)}\n\n")
#                     f.write(f"          Performance (ROC/AUC): {roc}\n\n")

#             # Write overall metrics and best results
#             # print(f"Overall Mean Accuracy: {overall_mean_accuracy}\n")
#             # print(f"Overall Variance of Accuracy: {overall_variance_accuracy}\n")
#             # print(f"Best Max Accuracy: {best_max_roc}\n")
#             print('\n')
#             info(f"Genes for Best Accuracy: {', '.join(best_genes)}\n")
#             info(f"Number of Genes for Best Accuracy: {best_num_genes}\n")
#             print('**' * 100 )



import torch
from lassonet import LassoNetClassifierCV, LassoNetRegressorCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from biomarl.feature_env import FeatureEvaluator
from biomarl.utils.logger import info

import numpy as np
import datetime
import argparse
import os
import pandas as pd

def gen_lassonet(fe: FeatureEvaluator, k=1000):
    x = fe.train.iloc[:, :-1].to_numpy()
    y = fe.train.iloc[:, -1].to_numpy()
    
    x = x.astype(float)
    if fe.task_type == 'cls':
        le = LabelEncoder()
        y = le.fit_transform(y)
        selector = LassoNetClassifierCV()
    else:
        selector = LassoNetRegressorCV()

    selector = selector.fit(x, y)
    scores = selector.feature_importances_
    value, indice = torch.topk(scores, k)
    choice = torch.zeros(x.shape[1])
    choice[indice] = 1

    test_result_roc, test_result_pr_auc, original_auc_roc, original_pr_auc = fe.report_performance(choice, flag='test', store=False)
    
    selected_features_indices = torch.nonzero(choice).squeeze().tolist()
    ranked_indices = [x for _, x in sorted(zip(scores[selected_features_indices], selected_features_indices), reverse=True)]
    
    return original_auc_roc, original_pr_auc, test_result_roc, test_result_pr_auc, choice, ranked_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run lassonet feature selection')
    parser.add_argument('--k', type=int, default=100, help='Number of top features to select')
    args = parser.parse_args()

    task_names = ['BRCA_ER', 'BRCA_HER2', 'BRCA_PR', 'BRCA_TN', 'LUAD_cls', 'OV_cls']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = 'results_v3'
    os.makedirs(base_results_dir, exist_ok=True)

    summary_file = os.path.join(base_results_dir, f'lassonet_all_results_summary_{timestamp}.txt')

    with open(summary_file, 'w') as summary_f:
        summary_f.write(f"LassoNet Feature Selection Summary - {datetime.datetime.now()}\n\n")

        for task_name in tqdm(task_names, desc="Processing tasks"):
            print(f"\n{'='*50}")
            print(f'Running lassonet feature selection for {task_name}')
            
            file_path = f'data/{task_name}.hdf'
            data = pd.read_hdf(file_path)
            gene_names = data.columns[:-1].tolist()
            
            all_results = []
            
            for run_idx in tqdm(range(1, 11), desc=f"Runs for {task_name}"):
                print(f'Run {run_idx}')
                fe = FeatureEvaluator(task_name)
                original_auc_roc, original_pr_auc, test_roc, test_pr, choice, ranked_indices = gen_lassonet(fe, k=args.k)
                
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
                # print(f"Selected genes: {len(ranked_genes)}")
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
            task_results_file = os.path.join(base_results_dir, f'lassonet_results_{task_name}_{timestamp}.txt')
            with open(task_results_file, 'w') as f:
                f.write(f'LassoNet results for {task_name}:\n\n')
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

            # Write summary for this task to the summary file 
            with open(summary_file, 'a') as summary_f:
                summary_f.write(f"\nResults for {task_name}:\n")
                summary_f.write(f"Original ROC: {np.mean(og_rocs):.4f} ± {np.std(og_rocs):.4f}\n")
                summary_f.write(f"Original PR: {np.mean(og_prs):.4f} ± {np.std(og_prs):.4f}\n")
                summary_f.write(f"Test ROC: {np.mean(test_rocs):.4f} ± {np.std(test_rocs):.4f}\n")
                summary_f.write(f"Test PR: {np.mean(test_prs):.4f} ± {np.std(test_prs):.4f}\n")

                #Add the performance lists section
                f.write('\n\n-- Performance Lists --\n')
                f.write(f'Original ROC Performances: {og_rocs}\n')
                f.write(f'Original PR Performances: {og_prs}\n')
                f.write(f'Test ROC Performances: {test_rocs}\n')
                f.write(f'Test PR Performances: {test_prs}\n')