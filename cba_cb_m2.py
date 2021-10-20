from collections import namedtuple
from csv import DictWriter
from functools import cmp_to_key
from typing import List

from utils import ruleitem
from utils.dataset import Dataset
from utils.rule import Rule
from utils.classifier_m2 import Classifier_m2


def is_satisfy(datacase:dict , rule:Rule) -> bool:
    """
    Check if datacase matches with rule's condset. If it matches, check if the rule's class label accurately predicts the datacase's class.
    """
    for item in rule.cond_set:
        if datacase[item] != rule.cond_set[item]:
            return None
    return True if datacase[-1] == rule.class_label else False


def compare_len(a,b):
    """
    Comparing length of rules
    Sort the set of generated rules car according to the relation ">", return the sorted rule list
    """
    return -1 if len(a.cond_set) < len(b.cond_set) else 0 if len(a.cond_set) == len(b.cond_set) else 1


def compare_support(a,b):
    """
    Comparing support of rules
    Sort the set of generated rules car according to the relation ">", return the sorted rule list
    """
    return 1 if a.support<b.support else compare_len(a,b) if a.support==b.support else -1


def comp_method(a,b):
    """
    Main compare method
    """
    return 1 if a.confidence<b.confidence else compare_support(a,b) if a.confidence==b.confidence else -1


def sort(car):
    """
    Sort rules based on the comp_method
    """
    def cmp_method(a,b):
        return comp_method(a,b)
    rule_list = list(car.rules)
    rule_list.sort(key=cmp_to_key(cmp_method))
    return rule_list


def compare(rule1, rule2) -> int:
    """
    Compare two rule, return the precedence.
    -1: rule1 < rule2, 0: rule1 < rule2 (randomly here), 1: rule1 > rule2
    """
    if rule1 is None and rule2 is not None:
        return -1
    elif rule1 is None and rule2 is None:
        return 0
    elif rule1 is not None and rule2 is None:
        return 1

    return -comp_method(rule1, rule2)


def sort_with_index(q, cars_list) -> set:
    """
    Sort the rule list order by precedence

    """
    def cmp_method(a, b):
        return comp_method(cars_list[a], cars_list[b])
    rule_list = list(q)
    rule_list.sort(key=cmp_to_key(cmp_method))
    return set(rule_list)


def ruleitem2rule(rule_item, dataset) -> Rule:
    """
    convert ruleitem of class RuleItem to rule of class Rule
    """
    rule = Rule(rule_item.cond_set, rule_item.class_label, dataset)
    return rule


def maxCoverRule(cars_list, data_case, boolean):
    """
    finds the highest precedence rule that covers the data case d from the set of rules having
    if boolean == True: same class as d
    if boolean == False: different class as d.
    """
    for i in range(len(cars_list)):
        if boolean == True:
            if cars_list[i].class_label == data_case[-1]:
                if is_satisfy(data_case, cars_list[i]):
                    return i
        else:
            if cars_list[i].class_label != data_case[-1]:
                temp_data_case = data_case[:-1]
                temp_data_case.append(cars_list[i].class_label)
                if is_satisfy(temp_data_case, cars_list[i]):
                    return i
    return None


def allCoverRules(u, data_case, c_rule, cars_list):
    """
    Finds all the rules in u that wrongly classify the data case and have higher precedences than that of its cRule.
    """
    w_set = set()
    for rule_index in u:
        # have higher precedences than cRule
        if compare(cars_list[rule_index], c_rule) > 0:
            # wrongly classify the data case
            if is_satisfy(data_case, cars_list[rule_index]) == False:
                w_set.add(rule_index)
    return w_set



def compClassDistr(dataset:Dataset) -> dict:
    """
    Counts the number of training cases in each class
    """
    from collections import Counter
    class_distr = dict()

    if len(dataset) <= 0:
        class_distr = None

    dataset_without_null = dataset
    while [] in dataset_without_null.data:
        dataset_without_null.data.remove([])

    class_column = dataset_without_null.get_class_list()
    class_distr = dict(Counter(class_column))
    return class_distr


def errorsOfRule(rule, dataset) -> int:
    """
    Get how many errors the rule wrongly classify the data case
    """
    error_number = 0
    for case in dataset:
        if case:
            if is_satisfy(case, rule) == False:
                error_number += 1
    return error_number


def selectDefault(class_distribution):
    """
    Choose the default class (majority class in remaining dataset)
    """
    if class_distribution is None:
        return None

    max = 0
    default_class = None
    for index in class_distribution:
        if class_distribution[index] > max:
            max = class_distribution[index]
            default_class = index
    return default_class


def defErr(default_class, class_distribution) -> int:
    """
    Count the number of errors that the default class will make in the remaining training data
    """
    if class_distribution is None:
        import sys
        return sys.maxsize

    error = 0
    for index in class_distribution:
        if index != default_class:
            error += class_distribution[index]
    return error


def classifier_builder_m2(cars, dataset:Dataset) -> Classifier_m2:
    """
    Main method of cba_cb_m2. Contains logic for Stage 1-3 to build classifier.
    """
    classifier = Classifier_m2()
    cars_list = sort(cars)
    for i in range(len(cars_list)):
        cars_list[i] = ruleitem2rule(cars_list[i], dataset)  # dataset

    # stage 1
    q = set()
    u = set()
    a = set()
    mark_set = set()
    A_item = namedtuple('A_item', ['id', 'class_label', 'cRule', 'wRule'])
    R_item = namedtuple('r_item', [ 'cRule', 'id', 'class_label'])


    for i in range(len(dataset)):
        c_rule_index = maxCoverRule(cars_list, dataset[i], True)
        w_rule_index = maxCoverRule(cars_list, dataset[i], False)

        if c_rule_index is not None:
            u.add(c_rule_index)
        if c_rule_index:
            cars_list[c_rule_index].classCasesCovered[dataset.get_label(i)] += 1
        if c_rule_index and w_rule_index:
            if compare(cars_list[c_rule_index], cars_list[w_rule_index]) > 0:
                q.add(c_rule_index)
                mark_set.add(c_rule_index)
            else:
                a_item = A_item(i, dataset.get_label(i), c_rule_index, w_rule_index )
                a.add(a_item)
        elif c_rule_index is None and w_rule_index is not None:
            a_item = A_item(i, dataset.get_label(i), c_rule_index, w_rule_index )
            a.add(a_item)

    # stage 2
    for entry in a:

        if cars_list[entry.wRule] in mark_set:
            if entry.cRule is not None:
                cars_list[entry.cRule].classCasesCovered[entry.class_label] -= 1
            cars_list[entry.wRule].classCasesCovered[entry.class_label] += 1
        else:
            if entry.cRule is not None:
                w_set = allCoverRules(u, dataset[entry.id], cars_list[entry.cRule], cars_list)
            else:
                w_set = allCoverRules(u, dataset[entry.id], None, cars_list)
            for w in w_set:
                r_item = R_item(entry.cRule, entry.id, entry.class_label)
                cars_list[w].replace.add(r_item)
                cars_list[w].classCasesCovered[entry.class_label] += 1
            q |= w_set

    # stage 3
    rule_errors = 0
    q = sort_with_index(q, cars_list)
    data_cases_covered = list([False] * len(dataset))
    for r_index in q:
        if cars_list[r_index].classCasesCovered[cars_list[r_index].class_label] != 0:
            for entry in cars_list[r_index].replace:
                if data_cases_covered[entry.id]:
                    cars_list[r_index].classCasesCovered[entry.class_label] -= 1
                else:
                    if entry.cRule is not None:
                        cars_list[entry.cRule].classCasesCovered[entry.class_label] -= 1
            for i in range(len(dataset)):
                datacase = dataset[i]

                if datacase:
                    is_satisfy_value = is_satisfy(datacase, cars_list[r_index])
                    if is_satisfy_value:
                        dataset[i] = []
                        data_cases_covered[i] = True
            rule_errors += errorsOfRule(cars_list[r_index], dataset)
            class_distribution = compClassDistr(dataset)
            default_class = selectDefault(class_distribution)
            default_errors = defErr(default_class, class_distribution)
            total_errors = rule_errors + default_errors
            classifier.add(cars_list[r_index], default_class, total_errors)
    classifier.discard()

    return classifier
