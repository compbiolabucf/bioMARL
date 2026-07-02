"""
feature env
interactive with the actor critic for the state and state after action
"""
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split

from biomarl.record import RecordList
from biomarl.utils.logger import error, info
from biomarl.utils.tools import test_task_new, downstream_task_new, downstream_task_by_method_std
from collections import defaultdict



# base_path = './data/paths_her2'
# base_path = './data/paths_pr2'
# base_path = './data/paths_tn'
# base_path = './data/paths_luad_cls'
# base_path = './data/paths_ov_cls'
base_path = os.environ.get('BIOMARL_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

# Define a function to return the default value
def default_value():
    return 'cls'

TASK_DICT = defaultdict(default_value,{'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg',
             'smtp': 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det', 'arrhythmia': 'cls',
             'nomao': 'cls', 'megawatt1': 'cls', 'activity':'mcls', 'mice_protein':'mcls', 'coil-20':'mcls', 'isolet':'mcls', 'minist':'mcls',
             'minist_fashion':'mcls', 'BRCA_ER':'cls', 'BRCA_HER2':'cls', 'BRCA_PR2':'cls', 'BRCA_TN':'cls', 'LUAD_mcls': 'mcls', 'OV_mcls': 'mcls', 
             'new_LUAD_mcls_kbest': 'mcls', 'new_OV_mcls_kbest': 'mcls', 'small_LUAD_mcls': 'mcls'
             })


MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc'],
    'reg': ['mae', 'mse', 'rae', 'rmse'],
    'det': ['map', 'f1_score', 'ras', 'recall'],
    'mcls' : ['precision', 'recall', 'mif1', 'maf1']
}

model_performance = {
    'mcls':namedtuple('ModelPerformance', MEASUREMENT['mcls']),
    'cls': namedtuple('ModelPerformance', MEASUREMENT['cls']),
    'reg': namedtuple('ModelPerformance', MEASUREMENT['reg']),
    'det': namedtuple('ModelPerformance', MEASUREMENT['det'])
}


class Evaluator(object):
    def __init__(self, task, task_type=None, dataset=None, method='RF', split=0.3, random_state = 42):
        self.original_report = None
        self.records = RecordList()
        self.task_name = task
        self.method = method
        self.split = split
        self.random_state = random_state
        if task_type is None:
            self.task_type = TASK_DICT[self.task_name]
        else:
            self.task_type = task_type

        # if dataset is None:
        #     data_path = os.path.join(base_path, self.task_name + '.hdf')
        #     original = pd.read_hdf(data_path)
        if dataset is None:
            hdf_path = os.path.join(base_path, self.task_name + '.hdf')
            h5_path = os.path.join(base_path, self.task_name + '.h5')

            if os.path.exists(hdf_path):
                data_path = hdf_path
            elif os.path.exists(h5_path):
                data_path = h5_path
            else:
                raise FileNotFoundError(f'H5 or hdf file not found')
            
            original = pd.read_hdf(data_path)
        else:
            original = dataset
        col = np.arange(original.shape[1])
        self.col_names = original.columns
        original.columns = col
        y = original.iloc[:, -1]
        x = original.iloc[:, :-1]
        if task == 'ap_omentum_ovary':
            y[y == 'Ovary'] = 1
            y[y == 'Omentum'] = 0
            y = y.astype(float)
            original = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
        self.original = original.fillna(value=0)
        y = self.original.iloc[:, -1]
        x = self.original.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.split,
                                                            shuffle=True, stratify= y, random_state= self.random_state)

        self.train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
        self.test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
        info('=' * 120)
        info('initializing the train and test dataset')
        self._check_path()

    def __len__(self):
        return len(self.records)

    def generate_data(self, operation, flag):
        pass

    def get_performance(self, data=None):
        if data is None:
            data = self.original
        return downstream_task_new(data, self.task_type)

    def report_ds(self):
        pass

    def _store_history(self, choice, performance_roc, performance_pr):
        self.records.append(choice, performance_roc, performance_pr)

    def _flush_history(self, choices, performances, is_permuted, num, padding):
        if is_permuted:
            flag_1 = 'augmented'
        else:
            flag_1 = 'original'
        if padding:
            flag_2 = 'padded'
        else:
            flag_2 = 'not_padded'
        torch.save(choices, f'{base_path}/history/{self.task_name}/choice.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the choice to {base_path}/history/{self.task_name}/choice.pt')
        torch.save(performances, f'{base_path}/history/{self.task_name}/performance.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the performance to {base_path}/history/{self.task_name}/performance.pt')

    def _check_path(self):
        if not os.path.exists(f'{base_path}/history/{self.task_name}'):
            os.makedirs(f'{base_path}/history/{self.task_name}', exist_ok=True)

    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:
            is_permuted = True
        else:
            is_permuted = False
        info('save the records...')
        choices, performances = \
            self.records.generate(num=num, padding=padding, padding_value=padding_value)
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_record(self, num=0, eos=-1):
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)

    def get_triple_record(self, num=0, eos=-1, mode='ht'):
        h_results = []
        labels = []
        t_results = []
        h_seed = []
        labels_seed = []
        for record in self.records.r_list:
            if mode.__contains__('h'):
                h, label = record.get_permutated(num, True, eos)
            else:
                h, label = record.repeat(num, True, eos)
            if mode.__contains__('t'):
                t, _ = record.get_permutated(num, True, eos)
            else:
                t, _ = record.repeat(num, True, eos)
            h_results.append(h)
            t_results.append(t)
            labels.append(label)
            h_seed.append(h_results[0])
            labels_seed.append(labels[0])
        return torch.cat(h_results, 0), torch.cat(labels, 0), torch.cat(t_results), \
               torch.cat(h_seed), torch.cat(labels_seed),

    def report_performance(self, choice, store=True, rp=True, flag=''):
        opt_ds = self.generate_data(choice, flag)
        # print(f'shape of opt_ds is: {opt_ds.shape}')
        a, b, c, d, e = test_task_new(opt_ds, task=self.task_type, method=self.method)
        report = model_performance[self.task_type](a, b, c, d, e)
        if flag == 'test':
            store = False
        if self.original_report is None:
            a, b, c, d, e = test_task_new(self.test, task=self.task_type, method=self.method)
            self.original_report = (a, b, c, d, e)
        else:
            a, b, c, d, e = self.original_report
        original_report = model_performance[self.task_type](a, b, c, d, e)
        # print("original Reported model performance 22:", original_report)
        if self.task_type == 'reg':
            final_result = report.rae
            if rp:
                info('1-MAE on original is: {:.4f}, 1-MAE on generated is: {:.4f}'.
                     format(original_report.mae, report.mae))
                info('1-MSE on original is: {:.4f}, 1-MSE on generated is: {:.4f}'.
                     format(original_report.mse, report.mse))
                info('1-RAE on original is: {:.4f}, 1-RAE on generated is: {:.4f}'.
                     format(original_report.rae, report.rae))
                info('1-RMSE on original is: {:.4f}, 1-RMSE on generated is: {:.4f}'.
                     format(original_report.rmse, report.rmse))
        elif self.task_type == 'cls':
            # final_result = report.f1_score
            final_result_roc = report.roc_auc
            final_result_pr_auc = report.pr_auc
            if rp:
                # info(f'\033[1;3mReporting performance for task: {self.task_name}\033[0m')
                info(f'Reporting performance for task: {self.task_name}')
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('F-1 on original is: {:.4f}, F-1 on generated is: {:.4f}'.
                     format(original_report.f1_score, report.f1_score))
                info('ROC/AUC on original is: {:.4f}, ROC/AUC on generated is: {:.4f}'.
                     format(original_report.roc_auc, report.roc_auc))
                info('PR-AUC on original is: {:.4f}, PR-AUC on generated is: {:.4f}'.
                     format(original_report.pr_auc, report.pr_auc))
        elif self.task_type == 'det':
            final_result = report.ras
            if rp:
                info(
                    'Average Precision Score on original is: {:.4f}, Average Precision Score on generated is: {:.4f}'
                    .format(original_report.map, report.map))
                info(
                    'F1 Score on original is: {:.4f}, F1 Score on generated is: {:.4f}'
                    .format(original_report.f1_score, report.f1_score))
                info(
                    'ROC AUC Score on original is: {:.4f}, ROC AUC Score on generated is: {:.4f}'
                    .format(original_report.ras, report.ras))
                info(
                    'Recall on original is: {:.4f}, Recall Score on generated is: {:.4f}'
                    .format(original_report.recall, report.recall))
        elif self.task_type == 'mcls':
            final_result = report.mif1
            if rp:
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('Micro-F1 on original is: {:.4f}, Micro-F1 on generated is: {:.4f}'.
                     format(original_report.mif1, report.mif1))
                info('Macro-F1 on original is: {:.4f}, Macro-F1 on generated is: {:.4f}'.
                     format(original_report.maf1, report.maf1))
        else:
            error('wrong task name!!!!!')
            assert False
        if store:
            if self.task_type == 'cls':
                self._store_history(choice, final_result_roc, final_result_pr_auc)
            else:
                self._store_history(choice, final_result)

        if self.task_type == 'cls':
            return final_result_roc, final_result_pr_auc, original_report.roc_auc, original_report.pr_auc
        elif self.task_type == 'mcls':
            return final_result, original_report.mif1
        else:
            return final_result


class FeatureEvaluator(Evaluator):
    def __init__(self, task, task_type=None, dataset=None, method='RF', split=0.3, random_state=42):
        # self.random_state = random_state
        super().__init__(task, task_type, dataset, method, split)
        self.ds_size = self.train.shape[1] - 1

    def generate_data(self, choice, flag=''):
        if choice.shape[0] != self.ds_size:
            error('wrong shape of choice')
            assert False
        if flag == 'test':
            ds = self.test
        elif flag == 'train':
            ds = self.train
        else:
            ds = self.original
        X = ds.iloc[:, :-1]
        indice = torch.arange(0, self.ds_size)[choice == 1]
        X = X.iloc[:, indice.tolist()].astype(np.float64)
        y = ds.iloc[:, -1].astype(np.float64)
        Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
        # info(f'shape of Dg is: {Dg.shape}')
        return Dg

    def _full_mask(self):
        return torch.FloatTensor([1] * self.ds_size)

    def report_ds(self):
        per = self.get_performance()
        #per = self.report_performance()
        info(f'current dataset : {self.task_name}')
        info(f'the size of shape is : {self.original.shape[1]}')
        info(f'original performance is : {per}')
        self._store_history(self._full_mask(), per)

if __name__ == '__main__':
    task_name = 'BRCA_ER'
    fe = FeatureEvaluator(task_name, method='RF')
    start_time = time.time()
    fe.report_performance()
    end_time = time.time()
    info(f'training on overall eval cost : {end_time - start_time}s')
    # for method in ['RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'DT', 'LASSO']:
    #     info(method)
    #     start_time = time.time()
    #     p, std = downstream_task_by_method_std(fe.original, fe.task_type, method)
    #     end_time = time.time()
    #     info(f'training on {method} eval cost : {end_time - start_time}s')
    # fe.report_performance(torch.FloatTensor([0, 1, 1, 0, 0]))
    # fe.save()
    # print(1)
