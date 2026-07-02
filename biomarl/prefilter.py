import numpy as np
import torch
from typing import List, Tuple, Union, Dict
import os
import pandas as pd
from biomarl.feature_env import FeatureEvaluator
from biomarl.scorers.kbest import *
from biomarl.scorers.svm import *
from biomarl.scorers.random_forest import *
from tqdm import tqdm


def normalize_scores(scores: np.ndarray, roc_perf: float) -> np.ndarray:
    """Min-max normalize a score vector to [0, 1], then weight it by the method's ROC-AUC."""
    scores = np.nan_to_num(scores, nan=0.0)
    if scores.max() - scores.min() != 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores * roc_perf


def calculate_normalized_weights(roc_performances: List[float]) -> np.ndarray:
    """Turn per-method ROC-AUC scores into weights that sum to 1."""
    performances = np.array(roc_performances)
    return performances / np.sum(performances)


def calculate_meta_scores(all_scores: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted aggregation of the per-method normalized score vectors."""
    meta_scores = np.zeros_like(all_scores[0])
    for scores, weight in zip(all_scores, weights):
        meta_scores += weight * scores
    return meta_scores

def get_pathway_dir(task_name: str) -> str:
   """Get pathway directory based on task name"""
   task_to_dir = {
       'BRCA_ER': 'paths_er',
       'BRCA_HER2': 'paths_her2', 
       'BRCA_PR': 'paths_pr',
       'BRCA_TN': 'paths_tn',
       'LUAD_cls': 'paths_luad_cls',
       'OV_cls': 'paths_ov_cls'
   }
   base_path = os.environ.get('BIOMARL_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
   return os.path.join(base_path, task_to_dir[task_name])

def load_pathway_datasets(pathway_dir: str) -> Dict:
   """Load all pathway datasets from directory"""
   pathway_data = {}
   for i in range(1, 187):  # 186 pathways
       file_path = os.path.join(pathway_dir, f'gene_cls_{i}.h5')
       if os.path.exists(file_path):
           data = pd.read_hdf(file_path)
           pathway_data[i] = data
   return pathway_data


def evaluate_pathway_performance(data: pd.DataFrame, task_name: str) -> float:
    """
    Evaluate pathway performance directly using RandomForestClassifier
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Use stratified split to maintain class distribution
    performances = []
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:,1]
        perf = roc_auc_score(y_test, y_pred_prob)
        performances.append(perf)

    return np.mean(performances)

def meta_feature_selection_with_pathways(fe: FeatureEvaluator, data: pd.DataFrame, k: int = 1000) -> Tuple[List[str], List[int], List[float]]:
    """Enhanced feature selection incorporating pathway information"""
    gene_names = data.columns[:-1].tolist()
    
    # Get pathway directory and load pathway datasets
    pathway_dir = get_pathway_dir(fe.task_name)
    pathway_data = load_pathway_datasets(pathway_dir)
    
    # Get pathway performances and track mapped genes
    pathway_performances = {}
    mapped_genes = set()
    
    for pathway_id, p_data in tqdm(pathway_data.items(), desc="Evaluating pathways"):
        pathway_score = evaluate_pathway_performance(p_data, fe.task_name)
        pathway_performances[pathway_id] = pathway_score
        mapped_genes.update(p_data.columns[:-1].tolist())
    
    # Get base scores for all genes using entire dataset
    kbest_result, _, kbest_scores = gen_kbest(fe, k=k)
    rf_result, _, rf_scores = gen_rf(fe, k=k)
    svm_result, _, svm_scores = gen_svm(fe, k=k)
    
    # Normalize scores based on performance
    performances = [kbest_result, rf_result, svm_result]
    normalized_scores = [
        normalize_scores(kbest_scores, kbest_result),
        normalize_scores(rf_scores, rf_result),
        normalize_scores(svm_scores, svm_result)
    ]
    
    # Calculate normalized weights
    weights = calculate_normalized_weights(performances)
    
    # Calculate meta-scores
    meta_scores = calculate_meta_scores(normalized_scores, weights)

    # base_features, base_indices, base_scores = meta_feature_selection(fe, k=k)
    
    # Create dictionary of base scores
    # all_scores = {gene_names[idx]: score for idx, score in zip(base_features, base_scores)}
    # all_scores = {gene_names[idx]: score for idx, score in zip(gene_names, meta_scores)}
    all_scores = dict(zip(gene_names, meta_scores))


    # For mapped genes, adjust scores based on pathway performance
    for gene in mapped_genes:
        if gene in all_scores:  # Check if gene was selected by meta_feature_selection
            # Find which pathways this gene belongs to
            gene_pathways = [pid for pid, p_data in pathway_data.items() 
                            if gene in p_data.columns[:-1].tolist()]
            
            if gene_pathways:
                pathway_bonus = np.mean([pathway_performances[pid] for pid in gene_pathways])
                # Use log scale for bonus to dampen effect
                all_scores[gene] *= (1 + np.log1p(pathway_bonus) * 0.05)  # Small bonus multiplier
                # all_scores[gene] += pathway_bonus * 0.1


    # Use mean + 2*std threshold
    scores_array = np.array(list(all_scores.values()))
    threshold = scores_array.mean() +  2 * scores_array.std()
    
    # Select features above threshold, up to k features
    selected_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    selected_items = [(g,s) for g,s in selected_items if s > threshold][:k]
    
    # Get gene names and indices
    print(f'len of scores is: {len(scores_array)}')
    print(f'Score statistics - Mean: {scores_array.mean():.4f}, Min: {scores_array.min():.4f}, Max: {scores_array.max():.4f}')    
    selected_features = [item[0] for item in selected_items]  # These are gene names
    # print(f'selected features are: {selected_features}')
    selected_indices = [gene_names.index(gene) for gene in selected_features]
    selected_scores = [item[1] for item in selected_items]
    # print(f'selected scores are: {selected_scores}')

    return selected_features, selected_indices, selected_scores

def filter_dataset_by_features(dataset: pd.DataFrame, selected_indices: List[int], gene_names: List[str]) -> pd.DataFrame:
   selected_columns = dataset.columns[selected_indices].tolist() + [dataset.columns[-1]]
   filtered_dataset = dataset[selected_columns]
   selected_gene_names = [gene_names[i] for i in selected_indices] + ['target']
   filtered_dataset.columns = selected_gene_names
   return filtered_dataset

def save_filtered_dataset(filtered_dataset: pd.DataFrame, save_path: str, key: str='data'):
   filtered_dataset.to_hdf(save_path, key=key, mode='w')

if __name__ == '__main__':
   import datetime
   import argparse
   
   print('=' * 100)
   print(f"Experimental results from {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   
   data_dir = "./data"
   result_dir = "./results"
   
   parser = argparse.ArgumentParser(description='prefilter data script')
   parser.add_argument('--task_name', type=str, required=True, help='Name of the task (e.g. BRCA_ER)')
   parser.add_argument('--k', type=int, default=1000, help='Number of top features to select')
   args = parser.parse_args()

   task_name = args.task_name
   file_path = os.path.join(data_dir, f'{task_name}.hdf')
   data = pd.read_hdf(file_path)
   gene_names = data.columns[:-1].tolist()
   
   print(f'Running meta feature selection for {task_name}')
           
   fe = FeatureEvaluator(task_name, split=0.3)
   selected_features, selected_indices, selected_scores = meta_feature_selection_with_pathways(fe, data, k=args.k)
   
   print("\nSelected Features:")
   for feature, score in zip(selected_features[:10], selected_scores[:10]):
       print(f"{feature}: {score:.4f}")

   filtered_dataset = filter_dataset_by_features(data, selected_indices, gene_names)
   save_path = f'./data/prefiltered_genes_{task_name}.hdf'
   save_filtered_dataset(filtered_dataset, save_path)
   
   print(f"\nTotal selected features: {len(selected_features)}")
   print('\n')
   print('*' * 100)