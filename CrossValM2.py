import random
import time
from absl import app

from read import read
from pre_processing import pre_process
from cba_rg import rule_generator
from cba_cb_m2 import classifier_builder_m2, is_satisfy

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
        self.ground_truth_labels = []
        self.pred_labels = []
        self.total_train = 0
        self.total_test = 0
        self.rules = set()
        self.num_folds = 10


    def get_error_rate(self, classifier, dataset, pred_labels):
        size = len(dataset)
        error_count = 0
        # print('--RULES--')
        # for idx, rule in enumerate(classifier.rule_list, start=1):
        #     print(f'rule{idx}')
        #     print(rule.__dict__)
        # print("default class", classifier.default_class)
        # print()
        # print('----')

        for case in dataset:  # case is e.g. [1, 1, 2, 2, 'Iris-versicolor']
            is_satisfy_value = False

            for rule in classifier.rule_list:
                # print('rule', rule.__dict__)

                is_satisfy_value = is_satisfy(case, rule, from_error=True)
                # print("is_satisfy_value", is_satisfy_value)
                if is_satisfy_value == True:
                    pred_labels.append(case[-1])
                    break
            # if is_satisfy_value == False:
            if not is_satisfy_value:

                if classifier.default_class != case[-1]:
                    # print('************************ ERROR *******************************')
                    pred_labels.append('wrong!')
                    error_count += 1
                else:
                    pred_labels.append(classifier.default_class)
        return error_count / size



    def create_train_test_ds(self, dataset, split, k):
        train_ds = dataset[:split[k]] + dataset[split[k+1]:]
        test_ds = dataset[split[k]:split[k+1]]
        return train_ds, test_ds


    def cross_validation(self):
        # read data
        data, attributes, value_type = read(self.data_path, self.scheme_path)
        random.Random(1).shuffle(data)
        dataset = pre_process(data, attributes, value_type)

        folds = int(len(dataset)/self.num_folds)
        split = [k*folds for k in range(0,self.num_folds)]
        split.append(len(dataset))

        self.ground_truth_labels = [data[-1] for data in dataset]

        for k in range(len(split)-1):
            print(f"========================== FOLD {k} ==========================")

            train_dataset, test_dataset = self.create_train_test_ds(dataset, split, k)
            self.total_train += len(train_dataset)
            self.total_test += len(test_dataset)

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








