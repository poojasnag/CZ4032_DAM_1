import time
from utils.dataset import Dataset

from utils.read import read
from utils.pre_processing import pre_process
from cba_rg import rule_generator
from cba_cb_m2 import classifier_builder_m2, is_satisfy
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

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
        self.total_prune_error_rate = 0
        self.pred_labels = []
        self.ground_truth = []
        self.rules = set()
        self.num_folds = 10


    def get_error_rate(self, classifier, dataset, pred_labels):
        error_count = 0
        for idx in range(len(dataset)): 
            is_satisfy_value = False
            for rule in classifier.rule_list:
                is_satisfy_value = is_satisfy(dataset[idx], rule)
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



    def create_train_test_ds(self, dataset, split, k):
        train_ds = dataset[:split[k]] + dataset[split[k+1]:]
        test_ds = dataset[split[k]:split[k+1]]
        return train_ds, test_ds

    # dictionary for multiple minsup values
    def class_minsup(self, dataset, multiple=False):
        # Store in dictionary with class as key and minsup as value
        actual_labels = dataset.get_class_list()
        class_freq = Counter(actual_labels)
        totalcount = len(actual_labels)
        for key, value in class_freq.items():
            if multiple:
                class_freq[key] = self.minsup * value/totalcount # minsup
            else:
                class_freq[key] = self.minsup
        class_freq = dict(class_freq)
        return class_freq


    def cross_validation(self, multiple=False, prune=False, dev=False):
        data, attributes, value_type = read(self.data_path, self.scheme_path)
        dataset = pre_process(data, attributes, value_type)

        kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

        k = 1
        for train_idx, test_idx in kf.split(dataset.get_values_list(),dataset.ground_truth_labels):
            print(f"========================== FOLD {k} ==========================")
            train_dataset = Dataset(dataset.get_indexes(train_idx), dataset.value_types, dataset.attributes)
            test_dataset = Dataset(dataset.get_indexes(test_idx), dataset.value_types, dataset.attributes)

            start_time = time.time()
            cars = rule_generator(train_dataset, self.class_minsup(dataset, multiple=multiple), self.minconf)


            if prune:
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

            if prune:
                print("CBA's error rate with pruning: %.1lf%%" % (error_rate * 100))
                print("No. of CARs with pruning: %d" % len(cars.rules))
                print("CBA-RG's run time with pruning: %.2lf s" % cba_rg_runtime)
                print("CBA-CB M2's run time with pruning: %.2lf s" % cba_cb_runtime)
                print("No. of rules in classifier of CBA-CB M2 with pruning: %d" % len(classifier.rule_list))
            else:
                print("CBA error rate without pruning: %.1lf%%" % (error_rate*100))
                print("No. of CARs without pruning: %d" % len(cars.rules))
                print("CBA-RG's run time without pruning: %.2lf s" % cba_rg_runtime)
                print("CBA-CB M2's run time without pruning: %.2lf s" % cba_cb_runtime)
                print("No. of rules in classifier of CBA-CB M2 without pruning: %d" % len(classifier.rule_list))

        if prune:
            print("\nAverage CBA's error rate with pruning: %.1lf%%" % (self.total_error_rate / 10 * 100))
            print("Average No. of CARs with pruning: %d" % int(self.total_car_num / 10))
            print("Average CBA-RG's run time with pruning: %.2lf s" % (self.cba_rg_total_runtime / 10))
            print("Average CBA-CB M2's run time with pruning: %.2lf s" % (self.cba_cb_total_runtime / 10))
            print("Average No. of rules in classifier of CBA-CB M2 with pruning: %d" % int(self.total_classifier_rule_num / 10))

        else:
            print("\nAverage CBA's error rate without pruning: %.1lf%%" % (self.total_error_rate / 10 * 100))
            print("Average No. of CARs without pruning: %d" % int(self.total_car_num / 10))
            print("Average CBA-RG's run time without pruning: %.2lf s" % (self.cba_rg_total_runtime / 10))
            print("Average CBA-CB M2's run time without pruning: %.2lf s" % (self.cba_cb_total_runtime / 10))
            print("Average No. of rules in classifier of CBA-CB M2 without pruning: %d" % int(self.total_classifier_rule_num / 10))
