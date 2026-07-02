import numpy as np
import pandas as pd
import os
import datetime
import mcdm
import argparse
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

from biomarl.feature_env import FeatureEvaluator
from biomarl.utils.logger import info

def get_feature_importances(estimator, getter='auto'):
    if getter == 'auto':
        if hasattr(estimator, 'coef_'):
            importances = estimator.coef_
        elif hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        else:
            raise ValueError("Estimator doesn't have coef_ or feature_importances_ attribute")
    else:
        importances = getattr(estimator, getter)
    
    return importances.reshape(-1)

def gen_mcdm(fe: FeatureEvaluator, k=1000):
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]

    # Convert object columns to numeric or categorical
    for col in x.select_dtypes(include=['object']).columns:
        try:
            x[col] = pd.to_numeric(x[col])
        except ValueError:
            x[col] = x[col].astype('category')
    
    if fe.task_type != 'reg' and not np.array_equal(np.unique(y), np.array([0, 1])):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    methods = ['RF', 'XGB', 'SVM', 'Ridge', 'LASSO', 'DT']
    # methods = ['RF']
    importances = []
    
    for method in methods:
        if fe.task_type == 'reg':
            if method == 'RF':
                model = RandomForestRegressor(random_state=0, n_jobs=1)
            elif method == 'XGB':
                model = XGBRegressor(eval_metric='logloss', n_jobs=1, enable_categorical = True)
            elif method == 'SVM':
                model = LinearSVR()
            elif method == 'Ridge':
                model = Ridge()
            elif method == 'LASSO':
                model = Ridge(alpha=1.0)
            else:  # DT
                model = DecisionTreeRegressor(max_depth=7, random_state=1)
        else:
            if method == 'RF':
                model = RandomForestClassifier(random_state=0, n_jobs=1)
            elif method == 'XGB':
                model = XGBClassifier(eval_metric='logloss', n_jobs=1, enable_categorical = True)
            elif method == 'SVM':
                model = LinearSVC()
            elif method == 'Ridge':
                model = RidgeClassifier()
            elif method == 'LASSO':
                model = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=1)
            else:  # DT
                model = DecisionTreeClassifier()
        
        selector = SelectFromModel(model)
        selector.fit(x, y)
        importance = get_feature_importances(selector.estimator_)
        importances.append(importance)
    
    importances = np.array(importances).T
    min_vals = importances.min(axis=0)
    max_vals = importances.max(axis=0)
    norm_importances = (importances - min_vals) / (max_vals - min_vals)

    if norm_importances.shape[1] < 2:
        # If only one method produced scores
        print("Warning: Not enough feature importance methods produced valid scores")
        # Simply rank by the single score
        ranked_indices = np.argsort(norm_importances.flatten())[::-1][:k]
    else:
        try:
            alt_names = [str(i) for i in range(x.shape[1])]
            rank = mcdm.rank(norm_importances, s_method="TOPSIS", n_method="Linear1",
                         c_method="AbsPearson", w_method="VIC", alt_names=alt_names)
            selected = rank[:k]
            ranked_indices = [int(i) for i, _ in selected]
        except Exception as e:
            print(f"MCDM ranking failed: {str(e)}")
            # Fallback to simple averaging of normalized scores
            avg_scores = np.mean(norm_importances, axis=1)
            ranked_indices = np.argsort(avg_scores)[::-1][:k]
    
    # Continue with rest of the function
    choice = np.zeros(fe.ds_size)
    choice[ranked_indices] = 1
    
    test_result_roc, test_result_pr_auc, original_auc_roc, original_pr_auc = fe.report_performance(choice, flag='test', store=False)
    
    # alt_names = [str(i) for i in range(x.shape[1])]
    # rank = mcdm.rank(norm_importances, s_method="TOPSIS", n_method="Linear1",
    #                  c_method="AbsPearson", w_method="VIC", alt_names=alt_names)
    
    # selected = rank[:k]
    # choice_index = [int(i) for i, _ in selected]
    # choice = np.zeros(fe.ds_size)
    # choice[choice_index] = 1
    
    # test_result_roc, test_result_pr_auc, original_auc_roc, original_pr_auc = fe.report_performance(choice, flag='test', store=False)
    
    return original_auc_roc, original_pr_auc, test_result_roc, test_result_pr_auc, choice, ranked_indices

def rest(X, y, task_type):
    if task_type == 'reg':
        return [i(X, y, task_type) for i in funcs]
    imps = []
    dep = 12
    for method in ['RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'DT']:
        if method == 'RF':
            if task_type == 'cls':
                model = RandomForestClassifier(random_state=0, n_jobs=1)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=1)
            else:
                model = RandomForestRegressor(max_depth=dep, random_state=0, n_jobs=1)
        elif method == 'XGB':
            if task_type == 'cls':
                model = XGBClassifier(eval_metric='logloss', n_jobs=1)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=1)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=2, n_jobs=128)
                # continue
                model = XGBRegressor(eval_metric='logloss', n_jobs=1)
        elif method == 'SVM':
            if task_type == 'cls':
                model = LinearSVC()
            elif task_type == 'mcls':
                model = LinearSVC()
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=3, n_jobs=128)
                # continue
                model = LinearSVR()
        elif method == 'Ridge':
            if task_type == 'cls':
                model = RidgeClassifier()
            elif task_type == 'mcls':
                model = OneVsRestClassifier(RidgeClassifier(), n_jobs=1)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=5, n_jobs=128)
                # continue
                model = Ridge()
        elif method == 'LASSO':
            if task_type == 'cls':
                model = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=1)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear'), n_jobs=1)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=8, n_jobs=128)
                model = DecisionTreeRegressor(max_depth=7, random_state=1)
                # continue
                # model = Lasso()
        else:  # dt
            if task_type == 'cls':
                model = DecisionTreeClassifier()
            elif task_type == 'mcls':
                model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=1)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=12, n_jobs=128)
                # continue
                model = DecisionTreeRegressor(max_depth=dep)
        #added by ehtesam       
        if not np.array_equal(np.unique(y), np.array([0, 1])):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        #ends here    
        selector = SelectFromModel(model)
        selector.fit(X, y)
        if task_type == 'mcls':
            if method!= 'SVM':
                overall_imp = []
                for i in selector.estimator_.estimators_:
                    overall_imp.append(get_feature_importances(i, getter='auto'))
                imps.append(np.concatenate([i.reshape(-1,1) for i in overall_imp], 1).mean(1))
            else:
                overall_imp = get_feature_importances(selector.estimator_, getter='auto')
                imps.append(overall_imp.mean(0))
        else:
            score = get_feature_importances(selector.estimator_, getter='auto')
            # if task_type == 'reg':
            #     score_ = torch.LongTensor(score.argsort())
            #     choice_index = score_[:k]
            #     choice = torch.zeros(score_.shape[0])
            #     choice[choice_index] = 1
            #     test_result = fe.report_performance(choice, flag='test', store=False, rp=False)
            #     print(f'{method}', choice_index, test_result)
            imps.append(score)
    return imps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run MCDM feature selection')
    parser.add_argument('--k', type=int, default=100, help='Number of top features to select')
    args = parser.parse_args()

    task_names = ['BRCA_ER', 'BRCA_HER2', 'BRCA_PR', 'BRCA_TN', 'LUAD_cls', 'OV_cls']
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = 'results_v3'
    os.makedirs(base_results_dir, exist_ok=True)
    
    summary_file = os.path.join(base_results_dir, f'mcdm_all_results_summary_{timestamp}.txt')
    
    with open(summary_file, 'w') as summary_f:
        summary_f.write(f"MCDM Feature Selection Summary - {datetime.datetime.now()}\n\n")
        
        for task_name in tqdm(task_names, desc="Processing tasks"):
            print(f"\n{'='*50}")
            print(f'Running MCDM feature selection for {task_name}')
            
            file_path = f'data/{task_name}.hdf'
            data = pd.read_hdf(file_path)
            gene_names = data.columns[:-1].tolist()
            
            all_results = []
            
            for run_idx in tqdm(range(1, 11), desc=f"Runs for {task_name}"):
                print(f'Run {run_idx}')
                fe = FeatureEvaluator(task_name)
                original_auc_roc, original_pr_auc, test_roc, test_pr, choice, ranked_indices = gen_mcdm(fe, k=args.k)
                
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
            task_results_file = os.path.join(base_results_dir, f'mcdm_results_{task_name}_{timestamp}.txt')
            with open(task_results_file, 'w') as f:
                f.write(f'MCDM results for {task_name}:\n\n')
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