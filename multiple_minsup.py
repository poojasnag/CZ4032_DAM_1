"""
Description: This is an improvement to the CBA algorithm as multiple minimum support (minsup) are used, 
instead of the usual just one minsup. The minsup controls how many rules and what kinds of rules are generated. 
1. If we set the minsup value too high, we may not find sufficient rules of infrequent classes. 
2. If we set the minsup value too low, we will find many useless and overfitting rules for frequent classes. 
Hence, for each class, a different minimum class support is assigned as follows: 
            minsup_of_class = total_minsup_constant * frequency_distribution_of_class

Referenced from: https://cgi.csc.liv.ac.uk/~frans/KDD/Software/CBA/cba.html 
"""

from collections import Counter

from dataset import *
from read import read
from pre_processing import pre_process
from cba_rg import *
from cba_cb_m2 import classifier_builder_m2, is_satisfy
import time
import random

# Using multiple minimum class supports 
# according to research paper, total_minsup is best at 0.01 
# def define_minsup(dataset, total_minsup, multiple): 

#     if multiple == True: 
#         actual_labels = dataset.get_class_list()
#         class_freq = Counter(actual_labels) 
#         totalcount = dataset.len()
#         for key, value in class_freq.items(): 
#             class_freq[key] = value/totalcount

#         for data in dataset: 
#             for key, value in class_freq.items(): 
#                 temp = data.data
#                 if temp[-1] == key: 
#                     minsup = total_minsup * value
#                     data.append(minsup) ## KIV incomplete change 

    # else: 
    #     for data in dataset: 
    #         data.append(total_minsup)
    
    # return dataset

# create class_freq that stores minsup for each class 
def class_minsup(dataset, total_minsup):
    # Store in dictionary with class as key and minsup as value 
    actual_labels = dataset.get_class_list()
    class_freq = Counter(actual_labels) 
    totalcount = len(actual_labels)
    for key, value in class_freq.items(): 
        class_freq[key] = total_minsup * value/totalcount # minsup 
    class_freq = dict(class_freq)
    return class_freq

# get unique minsup value for that class label 
def get_minsup(label, minsup_dict):
    minsup_value = minsup_dict.get(label)
    return minsup_value

# get the minimum minsup value 
def min_minsup(minsup_dict): 
    distinct = set(minsup_dict.values())
    return min(distinct)


# 10-fold cross-validations on CBA (M2) with pruning
## added in total_minsup and multiple to parameters 
def cross_validate_m2_with_prune2(data_path, scheme_path, total_minsup, multiple, minconf=0.5):
    data, attributes, value_type = read(data_path, scheme_path)
    # data is :  # (150, 5)
    # attributes is ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    # value type is ['numerical', 'numerical', 'numerical', 'numerical', 'label']

    random.Random(1).shuffle(data)
    dataset = pre_process(data, attributes, value_type)  # each entry is discretized [1, 2, 1, 1, 'Iris-setosa']
    
    # print(f"data {data}")
    block_size = int(len(dataset) / 10)
    split_point = [k * block_size for k in range(0, 10)]
    split_point.append(len(dataset))  # [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]

    cba_rg_total_runtime = 0
    cba_cb_total_runtime = 0
    total_car_number = 0
    total_classifier_rule_num = 0
    error_total_rate = 0
    ground_truth_labels = [data[-1] for data in dataset]
    pred_labels = []
    total_test = 0
    total_trng = 0
    rules = set()

    for k in range(len(split_point)-1):  # mini batch
        print("\nRound %d:" % k)

        training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
        test_dataset = dataset[split_point[k]:split_point[k+1]]
        total_test += len(test_dataset)
        total_trng += len(training_dataset)

        start_time = time.time()
        minsup = total_minsup
        cars = rule_generator(training_dataset, minsup, minconf)  # 135 trng items

        # print(cars.rules.pop().cond_set)

        cars.prune_rules(training_dataset)
        cars.rules = cars.pruned_rules
        end_time = time.time()
        cba_rg_runtime = end_time - start_time
        cba_rg_total_runtime += cba_rg_runtime

        start_time = time.time()
        classifier_m2 = classifier_builder_m2(cars, training_dataset)

        # print(classifier_m2.rule_list[0].__dict__)
        # {'cond_set': {0: 2, 3: 2}, 'class_label': 'Iris-versicolor', 'cond_sup_count': 21, 'rule_sup_count': 16, 'support': 0.11851851851851852, 'confidence': 0.7619047619047619, 'classCasesCovered': {'Iris-versicolor': 1, 'Iris-virginica': 2, 'Iris-setosa': 0}, 'replace': {(None, 6, 'Iris-virginica'), (None, 17, 'Iris-virginica')}}
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        cba_cb_total_runtime += cba_cb_runtime
        error_rate = get_error_rate(classifier_m2, test_dataset, pred_labels)  # float
        error_total_rate += error_rate

        total_car_number += len(cars.rules)
        total_classifier_rule_num += len(classifier_m2.rule_list)

        print("CBA's error rate with pruning: %.1lf%%" % (error_rate * 100))
        print("No. of CARs without pruning: %d" % len(cars.rules))
        print("CBA-RG's run time with pruning: %.2lf s" % cba_rg_runtime)
        print("CBA-CB M2's run time with pruning: %.2lf s" % cba_cb_runtime)
        print("No. of rules in classifier of CBA-CB M2 with pruning: %d" % len(classifier_m2.rule_list))

    print("\nAverage CBA's error rate with pruning: %.1lf%%" % (error_total_rate / 10 * 100))
    print("Average No. of CARs with pruning: %d" % int(total_car_number / 10))
    print("Average CBA-RG's run time with pruning: %.2lf s" % (cba_rg_total_runtime / 10))
    print("Average CBA-CB M2's run time with pruning: %.2lf s" % (cba_cb_total_runtime / 10))
    print("Average No. of rules in classifier of CBA-CB M2 with pruning: %d" % int(total_classifier_rule_num / 10))
    print('ground_truth: ', ground_truth_labels)
    print('\n\n')
    print('pred: ', pred_labels)
    print(len(pred_labels))
    # print('total_test', total_test)
    # print('total_trng', total_trng)

# if __name__ == "__main__":

#     test_data_path = 'datasets/iris.data'
#     test_scheme_path = 'datasets/iris.names'
#     total_minsup = 0.01
#     multiple = True

#     cross_validate_m2_with_prune(test_data_path, test_scheme_path, total_minsup, multiple)

# just for test
if __name__ == "__main__":
    # dataset1 = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
    #            [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    test_data = [
        ['red', 25.6, 56, 1],
        ['green', 33.3, 1, 1],
        ['green', 2.5, 23, 0],
        ['blue', 67.2, 111, 1],
        ['red', 29.0, 34, 0],
        ['yellow', 99.5, 78, 1],
        ['yellow', 10.2, 23, 1],
        ['yellow', 9.9, 30, 0],
        ['blue', 67.0, 47, 0],
        ['red', 41.8, 99, 1]
    ]
    test_attribute = ['color', 'average', 'age', 'class']
    test_value_type = ['categorical', 'numerical', 'numerical', 'label']
    test_data_after = pre_process(test_data, test_attribute, test_value_type)
    dataObj = Dataset(test_data_after, test_value_type, test_attribute)
    dataObj.num_attributes = len(dataObj.attributes)-1
    dataObj.ground_truth_labels = [d[-1] for d in dataObj.data]

    minsup = 0.15
    minconf = 0.6
    cars = rule_generator(dataObj, minsup, minconf)

    print("CARs:")
    cars.print_rule()

    print("prCARs:")
    cars.prune_rules(dataObj)
    cars.print_pruned_rule()
