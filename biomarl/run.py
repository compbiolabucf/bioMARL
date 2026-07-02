import os
import pandas as pd
import numpy as np
import torch
import argparse
import time
from multiprocessing import Pool, current_process
from datetime import datetime
import multiprocessing
import argparse

from biomarl.feature_env import FeatureEvaluator
from biomarl.model import gen_marlfs
from biomarl.prefilter import meta_feature_selection_with_pathways, filter_dataset_by_features, save_filtered_dataset
from biomarl.model import create_gene_mapping, create_pathway_embeddings, PathwayMetrics


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings('ignore')

# Paths default to repo-relative locations (repo root = parent of the biomarl/ package),
# overridable via env vars (BIOMARL_ROOT / BIOMARL_DATA_DIR / BIOMARL_RESULT_DIR / BIOMARL_PATHWAY_FILE).
_REPO_ROOT = os.environ.get('BIOMARL_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get('BIOMARL_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
PREFILTERED_DATA_DIR = DATA_DIR
RESULT_DIR = os.environ.get('BIOMARL_RESULT_DIR', os.path.join(_REPO_ROOT, 'results'))
PATHWAY_FILE_PATH = os.environ.get('BIOMARL_PATHWAY_FILE', os.path.join(_REPO_ROOT, 'pathways_new.txt'))

#ensure output directpries exists
os.makedirs(PREFILTERED_DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def prefilter_data(task_name, k):
    print(f'Prefiltering data for {task_name} with k={k}....')
    original_data_path = os.path.join(DATA_DIR, f'{task_name}.hdf')
    original_df = pd.read_hdf(original_data_path)
    gene_names = original_df.columns[:-1].tolist()

    fe = FeatureEvaluator(task_name)
    selected_features, selected_indices, selected_genes = meta_feature_selection_with_pathways(fe, original_df, k=k)

    filtered_df = filter_dataset_by_features(original_df, selected_indices, gene_names)
    prefiltered_save_path = os.path.join(PREFILTERED_DATA_DIR, f'prefiltered_genes_{task_name}.hdf')
    save_filtered_dataset(filtered_df, prefiltered_save_path)
    print(f'Prefiltered data saved to {prefiltered_save_path}')
    return prefiltered_save_path

def run_marlfs(task_config):
    prefiltered_data_path, task_name, pathway_file_path, gpu_id, run_idx, max_selected_features,\
                            explore_steps, use_meta, shared_memory, critic, way_2, n_samples = task_config
    print(f'running MARLFS for {task_name} (Run {run_idx}) on GPU {gpu_id}')
    torch.cuda.set_device(gpu_id)

    dataset = pd.read_hdf(prefiltered_data_path)
    gene_names = dataset.columns[:-1].tolist()
    # print(dataset.head(5))
    pathways = {}
    with open(pathway_file_path, 'r') as file:
        lines= file.readlines()
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                pathway_name = parts[0].strip()
                genes = parts[1].split(',')
                pathways[pathway_name] = genes

    gene_to_idx, filtered_pathways = create_gene_mapping(dataset=dataset, pathways=pathways)

    # print(f"Number of filtered pathways: {len(filtered_pathways)}")
    # print(f"Sample of filtered pathways: {list(filtered_pathways.items())[:1]}")

    pathway_embeddings = create_pathway_embeddings(filtered_pathways=filtered_pathways, n_genes = dataset.shape[1] - 1)
    pathway_metrics = PathwayMetrics(filtered_pathways, dataset.shape[1] -1)

    post_filter_task_name = f'prefiltered_genes_{task_name}'
    print(post_filter_task_name)
    fe = FeatureEvaluator(post_filter_task_name)

    try:
        (original_auc_roc, original_pr_auc, test_performance_roc, test_performance_pr_auc, optimal_set, num_features, 
            original_auc_roc_2, original_pr_auc_2, test_max_roc_2, test_max_pr_2,
            ranked_features, ranked_features_2, weighted_avg_values, improvements, all_states, 
            all_plot_action_values, dqn_list, all_rewards, q_values)= gen_marlfs(fe, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=explore_steps,
                                                                                  max_selected_features=max_selected_features, 
                                                                                  pathway_embeddings = pathway_embeddings,
                                                                                    pathway_metrics = pathway_metrics,
                                                                                    filtered_pathways=filtered_pathways,
                                                                                    use_meta=use_meta,
                                                                                    shared_memory=shared_memory,
                                                                                    critic=critic, n_samples = n_samples)
        
        ranked_gene_names = [gene_names[i] for i in ranked_features]
        ranked_gene_names_2 = [gene_names[i] for i in ranked_features_2]
        
        # return{
        #     'run_idx': run_idx,
        #     'gpu_id': gpu_id,
        #     'original_auc_roc': original_auc_roc,
        #     'original_pr_auc': original_pr_auc,
        #     'performance_roc': test_roc,
        #     'performance_pr': test_pr,
        #     'ranked_gene_names': ranked_gene_names
        # }

        return{
            'run_idx': run_idx,
            'gpu_id': gpu_id,
            'original_auc_roc': original_auc_roc,
            'original_pr_auc': original_pr_auc,
            'test_performance_roc': test_performance_roc,
            'test_performance_pr_auc': test_performance_pr_auc,
            'original_auc_roc_2': original_auc_roc_2 if way_2 else None,
            'original_pr_auc_2': original_pr_auc_2 if way_2 else None,
            'test_max_roc_2': test_max_roc_2 if way_2 else None,
            'test_max_pr_2': test_max_pr_2 if way_2 else None,
            'ranked_gene_names': ranked_gene_names,
            'ranked_gene_names_2': ranked_gene_names_2 if way_2 else None
            # 'error': str(e) if 'e' in locals() else None
        }
    except Exception as e:
        print(f'Error running MARLFS for {task_name} on GPU {gpu_id}: {e}')
        print(f'Full error traceback:')
        import traceback
        traceback.print_exc()
        return{
            'run_idx': run_idx,
            'gpu_id': gpu_id,
            'error': str(e)
        }
    
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    print(f'Experiment started at {datetime.now()}')
    start_time = time.time()

    parser = argparse.ArgumentParser(description='run feature selection pipeline')
    parser.add_argument('--task_name', type=str, required=True, help='name of the task (e.g BRCA_ER)')
    parser.add_argument('--use_meta', action='store_true', help='whether to use the surrogate model for reward calculation')
    parser.add_argument('--prefilter_num_feats', type=int, default=1000, help='number of features to select during prefiltering')
    parser.add_argument('--marlfs_max_feats', type=int, default=100, help='maximum number of selected features in marlfs')
    parser.add_argument('--marlfs_explore_steps', type=int, default=3000, help='number of exploration steps in marlfs')
    parser.add_argument('--num_marlfs_runs', type=int, default=10, help='number of marlfs runs')
    parser.add_argument('--shared_memory', action='store_true', help='Whether to use shared memory')
    parser.add_argument('--critic', action='store_true', help='Whether to use critic')
    parser.add_argument('--way_2', action='store_true', help='Whether to use second way of ranking')
    parser.add_argument('--n_samples', type=int, default=1500, help='number of samples for meta learner training')
    
    args = parser.parse_args()

    task_name = args.task_name
    num_marlfs_runs = args.num_marlfs_runs
    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {num_gpus}')

    all_marlfs_results = []

    print(f'starting processing for task: {task_name}')
    prefiltered_data_path = prefilter_data(task_name, k=args.prefilter_num_feats)

    marlfs_tasks = []
    for i in range(num_marlfs_runs):
        gpu_id = i % num_gpus
        marlfs_tasks.append((prefiltered_data_path, 
                           task_name, PATHWAY_FILE_PATH,
                           gpu_id, i,
                           args.marlfs_max_feats,
                           args.marlfs_explore_steps,
                           args.use_meta,
                           args.shared_memory,
                           args.critic,
                           args.way_2,
                           args.n_samples))

    print(f'running MARLFS for {task_name} with {num_marlfs_runs} runs on {num_gpus} GPUs.....')

    with Pool(processes=num_gpus) as pool:
        for result in pool.imap_unordered(run_marlfs, marlfs_tasks):
            all_marlfs_results.append(result)

        pool.close()
        pool.join()
        torch.cuda.empty_cache()

    results_file_path = os.path.join(RESULT_DIR, f"marlfs_results_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(results_file_path, 'w') as f:
        f.write(f'MARLFS results for {task_name}: \n')
        og_performances_roc = []
        og_performances_pr = []
        new_performances_roc = []
        new_performances_pr = []
        
        if args.way_2:
            og_performances_roc_2 = []
            og_performances_pr_2 = []
            new_performances_roc_2 = []
            new_performances_pr_2 = []

        for result in all_marlfs_results:
            if 'error' not in result:
                f.write(f"Run {result['run_idx']} (GPU {result['gpu_id']}):\n")
                f.write(f"  Original ROC AUC: {result['original_auc_roc']:.4f}\n")
                f.write(f"  Original PR AUC: {result['original_pr_auc']:.4f}\n")
                f.write(f"  Test ROC AUC: {result['test_performance_roc']:.4f}\n")
                f.write(f"  Test PR AUC: {result['test_performance_pr_auc']:.4f}\n")
                f.write(f"  Ranked Genes: {result['ranked_gene_names']}\n")
                
                og_performances_roc.append(result['original_auc_roc'])
                og_performances_pr.append(result['original_pr_auc'])
                new_performances_roc.append(result['test_performance_roc'])
                new_performances_pr.append(result['test_performance_pr_auc'])

                if args.way_2:
                    f.write(f"\n  Way 2 Results:\n")
                    f.write(f"  Original ROC AUC: {result['original_auc_roc_2']:.4f}\n")
                    f.write(f"  Test ROC AUC: {result['test_max_roc_2']:.4f}\n")
                    f.write(f"  Test PR AUC: {result['test_max_pr_2']:.4f}\n")
                    f.write(f"  Ranked Genes Way 2: {result['ranked_gene_names_2']}\n")

                    og_performances_roc_2.append(result['original_auc_roc_2'])
                    new_performances_roc_2.append(result['test_max_roc_2'])
                    new_performances_pr_2.append(result['test_max_pr_2'])
                
                f.write("\n")
            else:
                f.write(f"Run {result['run_idx']} (GPU {result['gpu_id']}): Error - {result['error']}\n\n")

        
        f.write('\n-- Performance Lists --\n')
        f.write(f'Original ROC Performances: {og_performances_roc}\n')
        f.write(f'Original PR Performances: {og_performances_pr}\n')
        f.write(f'Test ROC Performances: {new_performances_roc}\n')
        f.write(f'Test PR Performances: {new_performances_pr}\n')
        
        if args.way_2:
            f.write(f'\nWay 2 ROC Performances: {new_performances_roc_2}\n')
            f.write(f'Way 2 PR Performances: {new_performances_pr_2}\n')

        f.write('\n')

        f.write('\n-- Performance Summary Way 1 --\n')
        if og_performances_roc:
            original_mean_roc = np.mean(og_performances_roc)
            original_std_roc = np.std(og_performances_roc)
            original_mean_pr = np.mean(og_performances_pr)
            original_std_pr = np.std(og_performances_pr)
            f.write(f"Mean ± SD of Original ROC Performance: {original_mean_roc:.4f} ± {original_std_roc:.4f}\n")
            f.write(f"Mean ± SD of Original PR Performance: {original_mean_pr:.4f} ± {original_std_pr:.4f}\n")
        else:
            f.write('No successful original performances runs\n')

        if new_performances_roc:
            new_mean_roc = np.mean(new_performances_roc)
            new_std_roc = np.std(new_performances_roc)
            new_mean_pr = np.mean(new_performances_pr)
            new_std_pr = np.std(new_performances_pr)
            f.write(f"Mean ± SD of Test ROC Performance: {new_mean_roc:.4f} ± {new_std_roc:.4f}\n")
            f.write(f"Mean ± SD of Test PR Performance: {new_mean_pr:.4f} ± {new_std_pr:.4f}\n")
        else:
            f.write("No successful test performance runs.\n")

        if args.way_2:
            f.write('\n-- Performance Summary Way 2 --\n')
            if og_performances_roc_2:
                original_mean_roc_2 = np.mean(og_performances_roc_2)
                original_std_roc_2 = np.std(og_performances_roc_2)
                f.write(f"Mean ± SD of Original ROC Performance: {original_mean_roc_2:.4f} ± {original_std_roc_2:.4f}\n")
            else:
                f.write('No successful original performances runs for Way 2\n')

            if new_performances_roc_2:
                new_mean_roc_2 = np.mean(new_performances_roc_2)
                new_std_roc_2 = np.std(new_performances_roc_2)
                new_mean_pr_2 = np.mean(new_performances_pr_2)
                new_std_pr_2 = np.std(new_performances_pr_2)
                f.write(f"Mean ± SD of Test ROC Performance Way 2: {new_mean_roc_2:.4f} ± {new_std_roc_2:.4f}\n")
                f.write(f"Mean ± SD of Test PR Performance Way 2: {new_mean_pr_2:.4f} ± {new_std_pr_2:.4f}\n")
            else:
                f.write("No successful test performance runs for Way 2.\n")

    print(f'MARLFS results for {task_name} saved to {results_file_path}')

    print('All tasks completed.')
    end_time = time.time()
    total_time_spent = end_time - start_time  # Fixed the minus sign here
    print(f'Total time spent: {total_time_spent:.2f} seconds')
