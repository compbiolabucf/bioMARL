from typing import List
import torch
import torch.nn.functional as F
import numpy

# class Record(object):
#     def __init__(self, operation, performance):
#         if isinstance(operation, List):
#             self.operation = numpy.array(operation)
#         elif isinstance(operation, torch.Tensor):
#             self.operation = operation.numpy()
#         else:
#             assert isinstance(operation, numpy.ndarray)
#             self.operation = operation
#         self.performance = performance

#     def get_permutated(self):
#         pass

#     def get_ordered(self):
#         pass

#     def repeat(self):
#         pass

#     def __eq__(self, other):
#         if not isinstance(other, Record):
#             return False
#         return self.__hash__() == other.__hash__()

#     def __hash__(self):
#         return str(self.operation).__hash__()
    
class Record(object):
    def __init__(self, operation, roc_performance, pr_performance):
        if isinstance(operation, List):
            self.operation = numpy.array(operation)
        elif isinstance(operation, torch.Tensor):
            self.operation = operation.numpy()
        else:
            assert isinstance(operation, numpy.ndarray)
            self.operation = operation
        self.roc_performance = roc_performance
        self.pr_performance = pr_performance

    def get_permutated(self):
        pass

    def get_ordered(self):
        pass

    def repeat(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return str(self.operation).__hash__()

class SelectionRecord(Record):
    def __init__(self, operation, roc_performance, pr_performance):
        super().__init__(operation, roc_performance, pr_performance)  # Make sure parent Record class is updated too
        self.max_size = operation.shape[0]
        self.roc_performance = roc_performance
        self.pr_performance = pr_performance

    def _get_ordered(self):
        indice_select = torch.arange(0, self.max_size)[self.operation == 1]
        return indice_select, torch.FloatTensor([self.roc_performance]), torch.FloatTensor([self.pr_performance])

    def get_permutated(self, num=25, padding=True, padding_value=-1):
        ordered, roc_performance, pr_performance = self._get_ordered()
        size = ordered.shape[0]
        shuffled_indices = torch.empty(num + 1, size)
        shuffled_indices[0] = ordered
        roc_label = roc_performance.unsqueeze(0).repeat(num + 1, 1)
        pr_label = pr_performance.unsqueeze(0).repeat(num + 1, 1)
        for i in range(num):
            shuffled_indices[i + 1] = ordered[torch.randperm(size)]
        if padding and size < self.max_size:
            shuffled_indices = F.pad(shuffled_indices, (0, (self.max_size - size)), 'constant', padding_value)
        return shuffled_indices, roc_label, pr_label

    def repeat(self, num=25, padding=True, padding_value=-1):
        ordered, roc_performance, pr_performance = self._get_ordered()
        size = ordered.shape[0]
        roc_label = roc_performance.unsqueeze(0).repeat(num + 1, 1)
        pr_label = pr_performance.unsqueeze(0).repeat(num + 1, 1)
        indices = ordered.unsqueeze(0).repeat(num + 1, 1)
        if padding and size < self.max_size:
            indices = F.pad(indices, (0, (self.max_size - size)), 'constant', padding_value)
        return indices, roc_label, pr_label
    
class RecordList(object):
    def __init__(self):
        self.r_list = set()

    def append(self, op, roc_val, pr_val):
        self.r_list.add(SelectionRecord(op, roc_val, pr_val))

    def __len__(self):
        return len(self.r_list)

    def generate(self, num=25, padding=True, padding_value=-1):
        results = []
        roc_labels = []
        pr_labels = []
        for record in self.r_list:
            result, roc_label, pr_label = record.get_permutated(num, padding, padding_value)
            results.append(result)
            roc_labels.append(roc_label)
            pr_labels.append(pr_label)
        return torch.cat(results, 0), torch.cat(roc_labels, 0), torch.cat(pr_labels, 0)
    

# class SelectionRecord(Record):
#     def __init__(self, operation, performance):
#         super().__init__(operation, performance)
#         self.max_size = operation.shape[0]

#     def _get_ordered(self):
#         indice_select = torch.arange(0, self.max_size)[self.operation == 1]
#         return indice_select, torch.FloatTensor([self.performance])

#     def get_permutated(self, num=25, padding=True, padding_value=-1):
#         ordered, performance = self._get_ordered()
#         size = ordered.shape[0]
#         shuffled_indices = torch.empty(num + 1, size)
#         shuffled_indices[0] = ordered
#         label = performance.unsqueeze(0).repeat(num + 1, 1)
#         for i in range(num):
#             shuffled_indices[i + 1] = ordered[torch.randperm(size)]
#         if padding and size < self.max_size:
#             shuffled_indices = F.pad(shuffled_indices, (0, (self.max_size - size)), 'constant', padding_value)
#         return shuffled_indices, label

#     def repeat(self, num=25, padding=True, padding_value=-1):
#         ordered, performance = self._get_ordered()
#         size = ordered.shape[0]
#         label = performance.unsqueeze(0).repeat(num + 1, 1)
#         indices = ordered.unsqueeze(0).repeat(num + 1, 1)
#         if padding and size < self.max_size:
#             indices = F.pad(indices, (0, (self.max_size - size)), 'constant', padding_value)
#         return indices, label


# class RecordList(object):
#     def __init__(self):
#         self.r_list = set()

#     def append(self, op, val):
#         self.r_list.add(SelectionRecord(op, val))

#     def __len__(self):
#         return len(self.r_list)

#     def generate(self, num=25, padding=True, padding_value=-1):
#         results = []
#         labels = []
#         for record in self.r_list:
#             result, label = record.get_permutated(num, padding, padding_value)
#             results.append(result)
#             labels.append(label)

#         return torch.cat(results, 0), torch.cat(labels, 0)

