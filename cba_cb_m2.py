"""
Description: The following code implements an improved version of the algorithm, called CBA-CB: M2. It contains three
    stages. For stage 1, we scan the whole database, to find the cRule and wRule, get the set Q, U and A at the same
    time. In stage 2, for each case d that we could not decide which rule should cover it in stage 1, we go through d
    again to find all rules that classify it wrongly and have a higher precedence than the corresponding cRule of d.
    Finally, in stage 3, we choose the final set of rules to form our final classifer.
Input: a set of CARs generated from rule_generator (see cab_rg.py) and a dataset got from pre_process
    (see pre_processing.py)
Output: a classifier
Author: CBA Studio
"""
from collections import namedtuple
from functools import cmp_to_key

import ruleitem
from rule import Rule
from classifier_m2 import Classifier_m2


def is_satisfy(datacase, rule):
    for item in rule.cond_set:  # item = key of condset
        if datacase[item] != rule.cond_set[item]:  # check if datacase values match that of the rule's cond_set
            return None
    return True if datacase[-1] == rule.class_label else False

# sort the set of generated rules car according to the relation ">", return the sorted rule list
def compare_len(a,b):
    return -1 if len(a.cond_set) < len(b.cond_set) else 0 if len(a.cond_set) == len(b.cond_set) else 1

def compare_support(a,b):
    return 1 if a.support<b.support else compare_len(a,b) if a.support==b.support else -1

def comp_method(a,b):
    return 1 if a.confidence<b.confidence else compare_support(a,b) if a.confidence==b.confidence else -1

def sort(car):
    def cmp_method(a,b):
        return comp_method(a,b)
    rule_list = list(car.rules)
    rule_list.sort(key=cmp_to_key(cmp_method))
    return rule_list

# compare two rule, return the precedence.
#   -1: rule1 < rule2, 0: rule1 < rule2 (randomly here), 1: rule1 > rule2
def compare(rule1, rule2):
    if rule1 is None and rule2 is not None:
        return -1
    elif rule1 is None and rule2 is None:
        return 0
    elif rule1 is not None and rule2 is None:
        return 1

    return -comp_method(rule1, rule2)

# sort the rule list order by precedence
def sort_with_index(q, cars_list):
    def cmp_method(a, b):
        return comp_method(cars_list[a], cars_list[b])
    rule_list = list(q)
    rule_list.sort(key=cmp_to_key(cmp_method))
    return set(rule_list)


# convert ruleitem of class RuleItem to rule of class Rule
def ruleitem2rule(rule_item, dataset):
    rule = Rule(rule_item.cond_set, rule_item.class_label, dataset)
    return rule


# finds the highest precedence rule that covers the data case d from the set of rules having
#   if boolean == True: same class as d
#   if boolean == False: different class as d.
def maxCoverRule(cars_list, data_case, boolean):
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

# finds all the rules in u that wrongly classify the data case and have higher precedences than that of its cRule.
def allCoverRules(u, data_case, c_rule, cars_list):
    w_set = set()
    for rule_index in u:
        # have higher precedences than cRule
        if compare(cars_list[rule_index], c_rule) > 0:
            # wrongly classify the data case
            if is_satisfy(data_case, cars_list[rule_index]) == False:
                w_set.add(rule_index)
    return w_set



# counts the number of training cases in each class
def compClassDistr(dataset):
    from collections import Counter
    class_distr = dict()

    if len(dataset) <= 0:
        class_distr = None

    dataset_without_null = dataset
    while [] in dataset_without_null.data:
        dataset_without_null.data.remove([])

    class_column = dataset_without_null.get_class_list()
    class_distr = dict(Counter(class_column))
    # class_column = dataset_without_null.get_class_labels()
    # class_label = set(class_column)
    # for c in class_label:
    #     class_distr[c] = class_column.count(c)
    return class_distr


# get how many errors the rule wrongly classify the data case
def errorsOfRule(rule, dataset):
    error_number = 0
    for case in dataset:
        if case:
            if is_satisfy(case, rule) == False:
                error_number += 1
    return error_number


# choose the default class (majority class in remaining dataset)
def selectDefault(class_distribution):
    if class_distribution is None:
        return None

    max = 0
    default_class = None
    for index in class_distribution:
        if class_distribution[index] > max:
            max = class_distribution[index]
            default_class = index
    return default_class


# count the number of errors that the default class will make in the remaining training data
def defErr(default_class, class_distribution):
    if class_distribution is None:
        import sys
        return sys.maxsize

    error = 0
    for index in class_distribution:
        if index != default_class:
            error += class_distribution[index]
    return error


# main method, implement the whole classifier builder
def classifier_builder_m2(cars, dataset):
    """
    :param dataset: Dataset instance
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


# # just for test
# if __name__ == "__main__":
#     import cba_rg

#     dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
#                [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
#     minsup = 0.15
#     minconf = 0.6
#     cars = cba_rg.rule_generator(dataset, minsup, minconf)
#     classifier = classifier_builder_m2(cars, dataset)
#     classifier.print()

#     print()
#     dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
#                [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
#     cars.prune_rules(dataset)
#     cars.rules = cars.pruned_rules
#     classifier = classifier_builder_m2(cars, dataset)
#     classifier.print()