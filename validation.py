# """
# Description: This is our experimental code. We provide 4 test modes for experiments:
#     10-fold cross-validations on CBA (M1) without pruning
#     10-fold cross-validations on CBA (M1) with pruning
#     10-fold cross-validations on CBA (M2) without pruning
#     10-fold cross-validations on CBA (M2) with pruning
# Input: the relative directory path of data file and scheme file
# Output: the experimental results (similar to Table 1: Experiment Results in this paper)
# Author: CBA Studio
# """
# from read import read
# from pre_processing import pre_process
# from cba_rg import rule_generator
# from cba_cb_m2 import classifier_builder_m2, is_satisfy
# import time
# import random

# # calculate the error rate of the classifier on the dataset
# def get_error_rate(classifier, dataset, pred_labels=[]):
#     size = len(dataset)
#     error_number = 0
#     # print('-- RULES -- ')
#     # for idx, rule in enumerate(classifier.rule_list, start=1):
#         # print(f"rule {idx}")
#         # print(rule.__dict__)
#     # print("default class", classifier.default_class)
#     # print()
#     # print('----')

#     for case in dataset:  # case is e.g. [1, 1, 2, 2, 'Iris-versicolor']
#         is_satisfy_value = False

#         for rule in classifier.rule_list:
#             # print('rule', rule.__dict__)

#             is_satisfy_value = is_satisfy(case, rule, from_error=True)
#             # print("is_satisfy_value", is_satisfy_value)
#             if is_satisfy_value == True:
#                 pred_labels.append(case[-1])
#                 break
#         # if is_satisfy_value == False:
#         if not is_satisfy_value:

#             if classifier.default_class != case[-1]:
#                 print('************************ ERROR *******************************')
#                 pred_labels.append('wrong!')
#                 error_number += 1
#             else:
#                 pred_labels.append(classifier.default_class)
#     return error_number / size

# #######################################################################################################################################################

# ###                                                         USE THIS                                                                    ##############

# # 10-fold cross-validations on CBA (M2) with pruning
# def cross_validate_m2_with_prune(data_path, scheme_path, minsup=0.01, minconf=0.5):
#     data, attributes, value_type = read(data_path, scheme_path)
#     # data is :  # (150, 5)
#     # attributes is ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
#     # value type is ['numerical', 'numerical', 'numerical', 'numerical', 'label']

#     random.Random(1).shuffle(data)
#     dataset = pre_process(data, attributes, value_type)  # each entry is discretized [1, 2, 1, 1, 'Iris-setosa']
#     # print(f"data {data}")
#     block_size = int(len(dataset) / 10)
#     split_point = [k * block_size for k in range(0, 10)]
#     split_point.append(len(dataset))  # [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]

#     cba_rg_total_runtime = 0
#     cba_cb_total_runtime = 0
#     total_car_number = 0
#     total_classifier_rule_num = 0
#     error_total_rate = 0
#     ground_truth_labels = [data[-1] for data in dataset]
#     pred_labels = []
#     total_test = 0
#     total_trng = 0
#     rules = set()

#     for k in range(len(split_point)-1):  # mini batch
#         print("\nRound %d:" % k)

#         training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
#         test_dataset = dataset[split_point[k]:split_point[k+1]]
#         total_test += len(test_dataset)
#         total_trng += len(training_dataset)

#         start_time = time.time()
#         cars = rule_generator(training_dataset, minsup, minconf)  # 135 trng items

#         # print(cars.rules.pop().cond_set)

#         cars.prune_rules(training_dataset)
#         cars.rules = cars.pruned_rules
#         end_time = time.time()
#         cba_rg_runtime = end_time - start_time
#         cba_rg_total_runtime += cba_rg_runtime

#         start_time = time.time()
#         classifier_m2 = classifier_builder_m2(cars, training_dataset)

#         # print(classifier_m2.rule_list[0].__dict__)
#         # {'cond_set': {0: 2, 3: 2}, 'class_label': 'Iris-versicolor', 'cond_sup_count': 21, 'rule_sup_count': 16, 'support': 0.11851851851851852, 'confidence': 0.7619047619047619, 'classCasesCovered': {'Iris-versicolor': 1, 'Iris-virginica': 2, 'Iris-setosa': 0}, 'replace': {(None, 6, 'Iris-virginica'), (None, 17, 'Iris-virginica')}}
#         end_time = time.time()
#         cba_cb_runtime = end_time - start_time
#         cba_cb_total_runtime += cba_cb_runtime
#         error_rate = get_error_rate(classifier_m2, test_dataset, pred_labels)  # float
#         error_total_rate += error_rate

#         total_car_number += len(cars.rules)
#         total_classifier_rule_num += len(classifier_m2.rule_list)

#         print("CBA's error rate with pruning: %.1lf%%" % (error_rate * 100))
#         print("No. of CARs without pruning: %d" % len(cars.rules))
#         print("CBA-RG's run time with pruning: %.2lf s" % cba_rg_runtime)
#         print("CBA-CB M2's run time with pruning: %.2lf s" % cba_cb_runtime)
#         print("No. of rules in classifier of CBA-CB M2 with pruning: %d" % len(classifier_m2.rule_list))

#     print("\nAverage CBA's error rate with pruning: %.1lf%%" % (error_total_rate / 10 * 100))
#     print("Average No. of CARs with pruning: %d" % int(total_car_number / 10))
#     print("Average CBA-RG's run time with pruning: %.2lf s" % (cba_rg_total_runtime / 10))
#     print("Average CBA-CB M2's run time with pruning: %.2lf s" % (cba_cb_total_runtime / 10))
#     print("Average No. of rules in classifier of CBA-CB M2 with pruning: %d" % int(total_classifier_rule_num / 10))
#     # print('ground_truth: ', ground_truth_labels)
#     # print('\n\n')
#     # print('pred: ', pred_labels)
#     # print(len(pred_labels))


# # test entry goes here
# if __name__ == "__main__":
#     # using the relative path, all data sets are stored in datasets directory

#     test_data_path = 'datasets/iris.data'
#     test_scheme_path = 'datasets/iris.names'

#     # just choose one mode to experiment by removing one line comment and running
#     cross_validate_m2_with_prune(test_data_path, test_scheme_path)


#     # breast cancer - winsconsin
#     # classes: 2 for benign, 4 for malignant
