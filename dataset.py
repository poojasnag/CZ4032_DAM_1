from typing import List


class Dataset():
    def __init__(self, data, value_types, attributes):
        self.data = data  # [[0, 2, 0, 0, 'Iris-setosa'], ... ]
        self.value_types = value_types  # ['numerical', 'numerical', 'numerical', 'numerical', 'label']
        self.attributes = attributes  # ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        self.num_attributes = len(attributes) - 1
        self.ground_truth_labels = [d[-1] for d in self.data]

    def __setitem__(self, idx, item):
        self.data[idx] = item

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_label(self, idx):
        return self.data[idx][-1]

    def get_indexes(self, index_list):
        return [self.data[idx] for idx in index_list]

    def get_class_list(self):  # all class labels in dataset
        return [x[-1] for x in self.data]

    def get_distinct_values(self, column):
        return set([x[column] for x in self.data])

