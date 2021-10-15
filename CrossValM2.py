import random
import time
from absl import app
from dataset import Dataset

from read import read
from pre_processing import pre_process
from cba_rg import rule_generator
from cba_cb_m2 import classifier_builder_m2, is_satisfy

from sklearn.model_selection import KFold


class CrossValidationM2:
    def __init__(self, data_path, scheme_path, minsup, minconf):
        self.data_path = data_path
        self.scheme_path = scheme_path
        self.minsup = minsup
        self.minconf = minconf
        self.cba_rg_total_runtime = 0
        self.cba_cb_total_runtime = 0
        self.total_car_num = 0
        self.total_classifier_rule_num = 0
        self.total_error_rate = 0
        self.pred_labels = []
        # self.total_train = 0
        # self.total_test = 0
        self.rules = set()
        self.num_folds = 10


    def get_error_rate(self, classifier, dataset, pred_labels):
        error_count = 0
        # print('--RULES--')
        # for idx, rule in enumerate(classifier.rule_list, start=1):
        #     print(f'rule{idx}')
        #     print(rule.__dict__)
        # print("default class", classifier.default_class)
        # print()
        # print('----')

        for idx in range(len(dataset)):  # case is e.g. [1, 1, 2, 2, 'Iris-versicolor']
            is_satisfy_value = False
            for rule in classifier.rule_list:
                is_satisfy_value = is_satisfy(dataset[idx], rule, from_error=True)
                if is_satisfy_value == True:
                    pred_labels.append(dataset.get_label(idx))
                    break
            if not is_satisfy_value:
                if classifier.default_class != dataset.get_label(idx):
                    pred_labels.append('wrong!')
                    error_count += 1
                else:
                    pred_labels.append(classifier.default_class)
        return error_count / len(dataset)


    def cross_validation(self):
        # read data
        data, attributes, value_type = read(self.data_path, self.scheme_path)
        random.Random(1).shuffle(data)
        dataset = pre_process(data, attributes, value_type)

        kf = KFold(n_splits=10)

        k = 1
        for train_idx, test_idx in kf.split(dataset):
            print(f"========================== FOLD {k} ==========================")

            # train_dataset, test_dataset = self.create_train_test_ds(dataset, split, k)
            train_dataset = Dataset(dataset.get_indexes(train_idx), dataset.value_types, dataset.attributes)
            test_dataset = Dataset(dataset.get_indexes(test_idx), dataset.value_types, dataset.attributes)

            # self.total_train += len(train_dataset)
            # self.total_test += len(test_dataset)

            start_time = time.time()
            cars = rule_generator(train_dataset, self.minsup, self.minconf)

            # print(cars.rules.pop().condset)

            cars.prune_rules(train_dataset)
            cars.rules = cars.pruned_rules
            end_time = time.time()
            cba_rg_runtime = end_time-start_time
            self.cba_rg_total_runtime += cba_rg_runtime

            start_time = time.time()
            classifier = classifier_builder_m2(cars, train_dataset)
            end_time = time.time()
            cba_cb_runtime = end_time - start_time
            self.cba_cb_total_runtime += cba_cb_runtime

            error_rate = self.get_error_rate(classifier, test_dataset, self.pred_labels)
            self.total_error_rate += error_rate

            self.total_car_num += len(cars.rules)
            self.total_classifier_rule_num += len(classifier.rule_list)
            k += 1

            print("CBA error rate without pruning: %.1lf%%" % (error_rate*100))
            print("CBA's error rate with pruning: %.1lf%%" % (error_rate * 100))
            print("No. of CARs without pruning: %d" % len(cars.rules))
            print("CBA-RG's run time with pruning: %.2lf s" % cba_rg_runtime)
            print("CBA-CB M2's run time with pruning: %.2lf s" % cba_cb_runtime)
            print("No. of rules in classifier of CBA-CB M2 with pruning: %d" % len(classifier.rule_list))

        print("\nAverage CBA's error rate with pruning: %.1lf%%" % (self.total_error_rate / 10 * 100))
        print("Average No. of CARs with pruning: %d" % int(self.total_car_num / 10))
        print("Average CBA-RG's run time with pruning: %.2lf s" % (self.cba_rg_total_runtime / 10))
        print("Average CBA-CB M2's run time with pruning: %.2lf s" % (self.cba_cb_total_runtime / 10))
        print("Average No. of rules in classifier of CBA-CB M2 with pruning: %d" % int(self.total_classifier_rule_num / 10))
        # print('ground_truth: ', self.ground_truth_labels)
        # print('\n\n')
        # print('pred: ', self.pred_labels)
        # print(len(self.pred_labels))








