# # Genetic Feature Selection
# # Genetic algorithms in feature selection R.leardi 1996

# # from sklearn.feature_selection import GenericUnivariateSelect
# import pickle
# import torch
# from genetic_selection import GeneticSelectionCV
# from sklearn import preprocessing
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.svm import SVR, SVC

# from biomarl.feature_env import FeatureEvaluator
# from biomarl.utils.logger import info
# import random

# import numpy as np
# import datetime
# import os
# import pandas as pd
# from tqdm import tqdm

# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE,SIG_DFL)

# def gen_gfs(fe: FeatureEvaluator, limit_1, limit_2):
#     double_limit = True
#     while True:
#         results = []
#         x = fe.train.iloc[:, :-1].to_numpy()
#         y = fe.train.iloc[:, -1].to_numpy()
#         # print(fe.train.columns)
#         x_columns = x.shape[1]  # Get the number of columns in x
#         # print(x.columns)
#         k = max(2, int(random.uniform(limit_1 * x_columns, limit_2 * x_columns)))
#         if fe.task_type == 'reg':
#             estimator = SVR(kernel="linear")
#             # estimator = DecisionTreeRegressor()
#         else:
#             # estimator = SVC(kernel="linear")
#             # normalizer = preprocessing.Normalizer()
#             # normalizer.fit(x)
#             # x = normalizer.transform(x)
#             estimator = DecisionTreeClassifier()

#         selector = GeneticSelectionCV(
#             estimator,
#             n_jobs=-1,
#             max_features=k,
#             crossover_proba=0.5,
#             mutation_proba=0.2,
#             n_generations=30,
#             cv=5,
#             crossover_independent_proba=0.5,
#             mutation_independent_proba=0.05,
#             verbose = True
#         )

#         with tqdm(total = selector.n_generations, desc='Fitting Progress') as pbar:
#             selector = selector.fit(x,y)
#             pbar.update(1)
#         # selector = selector.fit(x,y)

#         choice = torch.FloatTensor(selector.support_)

#         if fe.task_type == 'cls':
#             test_result, original_result = fe.report_performance(choice, flag='test', store=False)
#             if (test_result - original_result) < -0.05 * original_result:
#                 if not double_limit:
#                     info(f'Performance improvement {test_result - original_result} is less than -5%, doubling limits and restarting the process.')
#                     limit_1 *= 1.5
#                     limit_2 *= 1.4
#                     double_limit = True
#                     continue
#                 else:
#                     info('Limits already doubled once, exiting the loop with current results.')
#                     break
#             else:
#                 break
#         else:
#             test_result = fe.report_performance(choice, flag='test', store=False)
#             original_result = fe.report_performance(torch.ones_like(choice), flag='test', store=False)
#             break

#     selected_features_indices = torch.nonzero(choice).squeeze().tolist()
#     # print(f'selected feature indices are: {len(selected_features_indices)}')
#     # feature_importances = selector.estimator_.feature_importances_ if hasattr(selector.estimator_, 'feature_importances_') else np.abs(selector.estimator_.coef_[0])
#     # print(f'feature importances are {feature_importances}')
#     # print(f'shape of feature importances is {feature_importances.shape}')

#     # Get feature importances
#     if hasattr(selector.estimator_, 'feature_importances_'):
#         feature_importances = selector.estimator_.feature_importances_
#     elif hasattr(selector.estimator_, 'coef_'):
#         feature_importances = np.abs(selector.estimator_.coef_[0])
#     else:
#         feature_importances = np.ones(x.shape[1])
#     # Create a dictionary mapping selected feature indices to their importances
#     feature_importance_dict = dict(zip(selected_features_indices, feature_importances))

    
#     # Sort the dictionary by importance (value) in descending order
#     sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
#     # Extract the sorted indices
#     ranked_indices = [idx for idx, _ in sorted_features]
#     ranked_features = [fe.train.columns[i] for i in ranked_indices]

#     info(f"The optimal performance is: {test_result}")
#     print(f'Optimal number of features selected: {int(choice.sum())}')
#     # print(f'Ranked features: {ranked_features}')
#     return original_result, test_result, choice, ranked_features

# if __name__ == '__main__':
#     print('=' * 100)
#     print(f"Experimental results from {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
#     data_dir = "data"
#     result_dir = "results"
#     # file_names = ["BRCA_ER.hdf", "BRCA_HER2.hdf", "BRCA_PR2.hdf", "BRCA_TN.hdf"]
#     # evals = ['Ridge', 'XGB', 'SVM', 'LASSO']
#     evals = ['KNN']
#     # file_names = ["BRCA_ER.hdf", "BRCA_HER2.hdf", "BRCA_PR2.hdf", "BRCA_TN.hdf"]
#     file_names = ['BRCA_TN.hdf']

#     for eval in evals:
#         for file_name in file_names:
#             task_name = file_name.split('.')[0]
#             file_path = os.path.join(data_dir, file_name)
#             data = pd.read_hdf(file_path)
#             gene_names = data.columns[:-1]
        
#             all_ranked_genes = []
#             roc_list = []
#             og_perf_list = []

#             best_max_roc = -1
#             best_genes = []
#             best_num_genes = 0
#             # eval = 'DT'

#             for run_idx in range(1, 11):
#                 fe = FeatureEvaluator(task_name, method=eval)
#                 print(f'Running feature selection for {task_name} with {eval}, run {run_idx}')
#                 og_result, test_result, choice, ranked_genes = gen_gfs(fe, limit_1=0.008, limit_2=0.014)
#                 ranked_genes = [gene_names[i] for i in ranked_genes]

#                 all_ranked_genes.append(ranked_genes)
#                 roc_list.append(test_result)
#                 og_perf_list.append(og_result)

#                 if test_result > best_max_roc:
#                     best_max_roc = test_result
#                     best_genes = ranked_genes
#                     best_num_genes = len(ranked_genes)

#             overall_mean_accuracy = np.mean(roc_list)
#             overall_variance_accuracy = np.var(roc_list)

#             info(f'All original results: {og_perf_list}')
#             info(f'Mean of all original results: {np.mean(og_perf_list)}')
#             info(f'Variance of all original performance: {np.var(og_perf_list)}')

#             print('\n')
#             info(f'All results: {roc_list}')
#             info(f"Overall Mean Accuracy with {eval} is : {overall_mean_accuracy}")
#             info(f"Overall Variance of Accuracy: {overall_variance_accuracy}")
#             info(f"Best Max Accuracy: {best_max_roc}")

#             result_file = os.path.join(result_dir, f'final_selection_gfs_{task_name}.txt')

#             with open(result_file, 'a') as f:
#                 f.write(f"Experimental results from {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
#                 for run_idx, (ranked_genes, roc) in enumerate(zip(all_ranked_genes, roc_list), 1):
#                     f.write(f"Run {run_idx}:\n")
#                     f.write(f"  Genes: {', '.join(ranked_genes)}\n")
#                     f.write(f"      Number of selected genes: {len(ranked_genes)}\n")
#                     f.write(f"          Performance (ROC/AUC): {roc}\n\n")

#             print('\n')
#             info(f"Genes for Best Accuracy: {', '.join(best_genes)}")
#             info(f"Number of Genes for Best Accuracy: {best_num_genes}")
#             print('**' * 100)

# import torch
# from genetic_selection import GeneticSelectionCV
# # from sklearn_genetic import GeneticSelectionCV
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.svm import SVR, SVC
# from tqdm import tqdm

# from biomarl.feature_env import FeatureEvaluator
# from biomarl.utils.logger import info

# import numpy as np
# import datetime
# import argparse
# import os
# import pandas as pd

# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE,SIG_DFL)

# def gen_gfs(fe: FeatureEvaluator, k=1000):
#     x = fe.train.iloc[:, :-1].to_numpy()
#     y = fe.train.iloc[:, -1].to_numpy()
    
    # if fe.task_type == 'reg':
    #     estimator = SVR(kernel="linear")
    # else:
    #     estimator = DecisionTreeClassifier()

#     selector = GeneticSelectionCV(
#         estimator,
#         cv=5,
#         verbose=1,
#         max_features=k,
#         n_population=50,
#         crossover_proba=0.5,
#         mutation_proba=0.2,
#         n_generations=30,
#         tournament_size=3,
#         n_gen_no_change=10,
#         n_jobs=-1,
#         fit_params=None
#     )

#     # selector = GeneticSelectionCV(
#     #     estimator,
#     #     cv=5,
#     #     verbose=1,
#     #     scoring="accuracy",
#     #     max_features=5,
#     #     n_population=50,
#     #     crossover_proba=0.5,
#     #     mutation_proba=0.2,
#     #     n_generations=40,
#     #     crossover_independent_proba=0.5,
#     #     mutation_independent_proba=0.05,
#     #     tournament_size=3,
#     #     n_gen_no_change=10,
#     #     caching=True,
#     #     n_jobs=-1,
#     # )

#     with tqdm(total=selector.n_generations, desc='Fitting Progress') as pbar:
#         selector = selector.fit(x, y)
#         pbar.update(1)

#     choice = torch.FloatTensor(selector.support_)
#     test_result_roc, test_result_pr_auc, original_auc_roc, original_pr_auc = fe.report_performance(choice, flag='test', store=False)
    
#     selected_features_indices = torch.nonzero(choice).squeeze().tolist()
    
#     if hasattr(selector.estimator_, 'feature_importances_'):
#         feature_importances = selector.estimator_.feature_importances_
#     elif hasattr(selector.estimator_, 'coef_'):
#         feature_importances = np.abs(selector.estimator_.coef_[0])
#     else:
#         feature_importances = np.ones(x.shape[1])
        
#     feature_importance_dict = dict(zip(selected_features_indices, feature_importances))
#     sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
#     ranked_indices = [idx for idx, _ in sorted_features]
    
#     return original_auc_roc, original_pr_auc, test_result_roc, test_result_pr_auc, choice, ranked_indices


import torch
# from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from tqdm import tqdm

from biomarl.feature_env import FeatureEvaluator
from biomarl.utils.logger import info

import numpy as np
import datetime
import argparse
import os
import pandas as pd

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from tqdm import tqdm
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import numpy as np

def gen_gfs(fe: FeatureEvaluator, k=1000):
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer, Continuous, Categorical
    x = fe.train.iloc[:, :-1].to_numpy()
    y = fe.train.iloc[:, -1].to_numpy()
    n_features = x.shape[1]
    
    # Use standard sklearn estimators
    if fe.task_type == 'reg':
        estimator = SVR(kernel="linear")
        param_grid = {"C": Integer(1, 10)}  # For SVR
    else:
        estimator = DecisionTreeClassifier()
        param_grid = {
            "max_depth": Integer(2, k),  # More reasonable max_depth range
            "min_samples_split": Integer(2, 20)   # Add another parameter for better optimization
        }
    # else:
    #     estimator = DecisionTreeClassifier()
    #     param_grid = {"max_depth": Integer(1, k)}

    selector = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        verbose=True,
        scoring="accuracy" if fe.task_type == "cls" else "neg_mean_squared_error",
        population_size=50,
        generations=30,
        crossover_probability=0.5,
        mutation_probability=0.2,
        n_jobs=-1
    )

    with tqdm(total=selector.generations, desc='Fitting Progress') as pbar:
        selector.fit(x, y)
        pbar.update(1)

    # selector.fit(x,y)

    # Get feature importances from trained model
    if hasattr(selector.best_estimator_, 'feature_importances_'):
        importances = selector.best_estimator_.feature_importances_
    elif hasattr(selector.best_estimator_, 'coef_'):
        importances = np.abs(selector.best_estimator_.coef_[0])
    else:
        importances = np.ones(n_features)
    
    # Select top k features based on importance
    indices = np.argsort(importances)[-k:]
    choice = torch.zeros(n_features)
    choice[indices] = 1
    
    test_result_roc, test_result_pr_auc, original_auc_roc, original_pr_auc = fe.report_performance(choice, flag='test', store=False)
    
    # Get ranked indices based on importance scores
    sorted_features = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)
    ranked_indices = [idx for idx, _ in sorted_features[:k]]
    
    return original_auc_roc, original_pr_auc, test_result_roc, test_result_pr_auc, choice, ranked_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run genetic feature selection')
    parser.add_argument('--k', type=int, default=100, help='Number of top features to select')
    args = parser.parse_args()

    task_names = ['BRCA_ER', 'BRCA_HER2', 'BRCA_PR', 'BRCA_TN', 'LUAD_cls', 'OV_cls']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = 'results_v3'
    os.makedirs(base_results_dir, exist_ok=True)

    summary_file = os.path.join(base_results_dir, f'gfs_all_results_summary_{timestamp}.txt')

    with open(summary_file, 'w') as summary_f:
        summary_f.write(f"Genetic Feature Selection Summary - {datetime.datetime.now()}\n\n")

        for task_name in tqdm(task_names, desc="Processing tasks"):
            print(f"\n{'='*50}")
            print(f'Running genetic feature selection for {task_name}')
            
            file_path = f'data/{task_name}.hdf'
            data = pd.read_hdf(file_path)
            gene_names = data.columns[:-1].tolist()
            
            all_results = []
            
            for run_idx in tqdm(range(1, 11), desc=f"Runs for {task_name}"):
                print(f'Run {run_idx}')
                fe = FeatureEvaluator(task_name)
                original_auc_roc, original_pr_auc, test_roc, test_pr, choice, ranked_indices = gen_gfs(fe, k=args.k)
                
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
            task_results_file = os.path.join(base_results_dir, f'gfs_results_{task_name}_{timestamp}.txt')
            with open(task_results_file, 'w') as f:
                f.write(f'Genetic Feature Selection results for {task_name}:\n\n')
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

                #Add the performance lists section
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
