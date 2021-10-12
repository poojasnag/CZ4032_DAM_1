"""
Description: The implementation of CBA-RG algorithm, generating the complete set of CARs (Class Association Rules).
    We just follow up algorithm raised up in the paper without improvement.
Input: a dataset got from pre_process (see pre_processing.py), minsup and minconf
Output: CARs
Author: CBA Studio
Reference: https://www.cs.uic.edu/~hxiao/courses/cs594-slides.pdf
"""
import ruleitem
import sys


class FrequentRuleitems:
    """
    A set of frequent k-ruleitems, just using set.
    """
    def __init__(self):
        self.frequent_ruleitems_set = set()

    # get size of set
    def get_size(self):
        return len(self.frequent_ruleitems_set)

    # add a new ruleitem into set
    def add(self, rule_item):
        is_existed = False
        for item in self.frequent_ruleitems_set:
            if item.class_label == rule_item.class_label:
                if item.cond_set == rule_item.cond_set:
                    is_existed = True
                    break
        if not is_existed:
            self.frequent_ruleitems_set.add(rule_item)

    # append set of ruleitems
    def append(self, sets):
        for item in sets.frequent_ruleitems:
            self.add(item)

    # print out all frequent ruleitems
    def print(self):
        for item in self.frequent_ruleitems_set:
            item.print()


class Car:
    """
    Class Association Rules (Car). If some ruleitems has the same condset, the ruleitem with the highest confidence is
    chosen as the Possible Rule (PR). If there're more than one ruleitem with the same highest confidence, we randomly
    select one ruleitem.
    """
    def __init__(self):
        self.rules = set()
        self.pruned_rules = set()

    # print out all rules
    def print_rule(self):
        for item in self.rules:
            item.print_rule()

    # print out all pruned rules
    def print_pruned_rule(self):
        for item in self.pruned_rules:
            item.print_rule()

    # add a new rule (frequent & accurate), save the ruleitem with the highest confidence when having the same condset
    def _add(self, rule_item, minsup, minconf):
        if rule_item.support >= minsup and rule_item.confidence >= minconf:
            if rule_item in self.rules:
                return
            for item in self.rules:
                if item.cond_set == rule_item.cond_set and item.confidence < rule_item.confidence:
                    self.rules.remove(item)
                    self.rules.add(rule_item)
                    return
                elif item.cond_set == rule_item.cond_set and item.confidence >= rule_item.confidence:
                    return
            self.rules.add(rule_item)

    # convert frequent ruleitems into car
    def gen_rules(self, frequent_ruleitems, minsup, minconf):
        for item in frequent_ruleitems.frequent_ruleitems_set:
            self._add(item, minsup, minconf)

    # prune rules
    def prune_rules(self, dataset):
        for rule in self.rules:
            # pruned_rule = prune(rule, dataset)  # return object
            pruner = Prune(rule, dataset)
            pruner.find_prune_rule(rule)
            pruned_rule = pruner.pruned_rule
            # print("pruned_rule", pruned_rule)
            # print("class_label", pruned_rule.class_label)

            is_existed = False
            for rule in self.pruned_rules:
                if rule.class_label == pruned_rule.class_label:
                    if rule.cond_set == pruned_rule.cond_set:
                        is_existed = True
                        break

            if not is_existed:
                self.pruned_rules.add(pruned_rule)

    # union new car into rules list
    def append(self, car, minsup, minconf):
        for item in car.rules:
            self._add(item, minsup, minconf)

class Prune:
    def __init__(self, initial_rule, dataset):
        self.dataset = dataset
        self.min_rule_error = sys.maxsize
        self.pruned_rule = initial_rule

    def errors_of_rule(self, r):  # input rule
        import cba_cb_m2

        errors_number = 0
        for case in self.dataset:
            if cba_cb_m2.is_satisfy(case, r) == False:
            # if not cba_cb_m2.is_satisfy(case, r):
                errors_number += 1
        return errors_number
  # prune rule recursively
    def find_prune_rule(self, this_rule):
        # calculate how many errors the rule r make in the dataset
        rule_error = self.errors_of_rule(this_rule)
        if rule_error < self.min_rule_error:
            self.min_rule_error = rule_error
            self.pruned_rule = this_rule
        this_rule_cond_set = list(this_rule.cond_set)
        if len(this_rule_cond_set) >= 2:
            for attribute in this_rule_cond_set:
                temp_cond_set = dict(this_rule.cond_set)
                temp_cond_set.pop(attribute)
                temp_rule = ruleitem.RuleItem(temp_cond_set, this_rule.class_label, self.dataset)
                temp_rule_error = self.errors_of_rule(temp_rule)
                if temp_rule_error <= self.min_rule_error:
                    self.min_rule_error = temp_rule_error
                    self.pruned_rule = temp_rule
                    if len(temp_cond_set) >= 2:
                        self.find_prune_rule(temp_rule)

# invoked by candidate_gen, join two items to generate candidate
def join(item1, item2, dataset):
    if item1.class_label != item2.class_label:
        return None
    category1 = item1.cond_set.items()  #set(item1.cond_set)
    category2 = item2.cond_set.items() # set(item2.cond_set)
    print('item1.cond_set', item1.cond_set)
    print('cat_1----------------------', category1)
    if category1 == category2:
        return None

    intersect = dict(category1 & category2)
    if not intersect: return None
    # intersect = category1 & category2
    # for item in intersect:
    #     if item1.cond_set[item] != item2.cond_set[item]:
    #         return None
    # category = category1 | category2
    new_cond_set = dict(category1 | category2)
    # new_cond_set = dict()
    # for item in category:
    #     if item in category1:
    #         new_cond_set[item] = item1.cond_set[item]
    #     else:
    #         new_cond_set[item] = item2.cond_set[item]
    new_ruleitem = ruleitem.RuleItem(new_cond_set, item1.class_label, dataset)
    return new_ruleitem

# TODO: replace this!!!
# similar to Apriori-gen in algorithm Apriori
def candidate_gen(frequent_ruleitems, dataset):
    returned_frequent_ruleitems = FrequentRuleitems()
    for item1 in frequent_ruleitems.frequent_ruleitems_set:
        for item2 in frequent_ruleitems.frequent_ruleitems_set:
            new_ruleitem = join(item1, item2, dataset)
            if new_ruleitem:
                returned_frequent_ruleitems.add(new_ruleitem)
                if returned_frequent_ruleitems.get_size() >= 1000:      # not allow to store more than 1000 ruleitems
                    return returned_frequent_ruleitems
    return returned_frequent_ruleitems

#################################################################### CBA-RG ######################################################################################

# main method, implementation of CBA-RG algorithm
def rule_generator(dataset, minsup, minconf):
    frequent_ruleitems = FrequentRuleitems()
    car = Car()

    # FIRST SCAN (C1)
    # TODO: Separate out this part
    # get large 1-ruleitems and generate rules
    class_label = set([x[-1] for x in dataset])
    for column in range(0, len(dataset[0])-1):  # range(4) each col is a feature
        distinct_value = set([x[column] for x in dataset])  # {1,2,3}
        for value in distinct_value:
            cond_set = {column: value}
            for classes in class_label:
                rule_item = ruleitem.RuleItem(cond_set, classes, dataset)
                if rule_item.support >= minsup:
                    frequent_ruleitems.add(rule_item)
    # L1
    car.gen_rules(frequent_ruleitems, minsup, minconf)
    cars = car

    # print(cars.rules.pop().__dict__)

    # {'cond_set': {1: 1}, 'class_label': 'Iris-versicolor', 'cond_sup_count': 72, 'rule_sup_count': 36, 'support': 0.26666666666666666, 'confidence': 0.5}

    last_cars_number = 0
    current_cars_number = len(cars.rules)
    while frequent_ruleitems.get_size() > 0 and current_cars_number <= 2000 and \
                    (current_cars_number - last_cars_number) >= 10:
        candidate = candidate_gen(frequent_ruleitems, dataset)
        frequent_ruleitems = FrequentRuleitems()
        car = Car()
        for item in candidate.frequent_ruleitems_set:
            if item.support >= minsup:
                frequent_ruleitems.add(item)
        car.gen_rules(frequent_ruleitems, minsup, minconf)
        cars.append(car, minsup, minconf)
        last_cars_number = current_cars_number
        current_cars_number = len(cars.rules)

    # print(cars.rules.pop().__dict__)
    # {'cond_set': {1: 2, 2: 1, 3: 1}, 'class_label': 'Iris-setosa', 'cond_sup_count': 18, 'rule_sup_count': 18, 'support': 0.13333333333333333, 'confidence': 1.0}

    return cars


# just for test
if __name__ == "__main__":
    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    minsup = 0.15
    minconf = 0.6
    cars = rule_generator(dataset, minsup, minconf)

    print("CARs:")
    cars.print_rule()

    print("prCARs:")
    cars.prune_rules(dataset)
    cars.print_pruned_rule()
