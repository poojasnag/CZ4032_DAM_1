from utils import ruleitem
from utils.pre_processing import *
from utils.frequentRuleItems import *
from utils.car import *

def join(item1: FrequentRuleitems, item2: FrequentRuleitems, dataset: Dataset) -> ruleitem.RuleItem:
    """
    Invoked by candidate_gen, join two items to generate candidate
    """
    if item1.class_label != item2.class_label:
        return None
    category1 = item1.cond_set.items()
    category2 = item2.cond_set.items()
    if category1 == category2:
        return None
    intersect = dict(category1 & category2)
    if set(intersect.keys()) == (dict(category1).keys() & dict(category2).keys()):
        new_cond_set = dict(category1 | category2)
        new_ruleitem = ruleitem.RuleItem(new_cond_set, item1.class_label, dataset)
        return new_ruleitem
    return None

def candidate_gen(frequent_ruleitems: FrequentRuleitems, dataset: Dataset) -> FrequentRuleitems:
    """
    Similar to Apriori-gen in algorithm Apriori
    """
    returned_frequent_ruleitems = FrequentRuleitems()
    for item1 in frequent_ruleitems.frequent_ruleitems_set:
        for item2 in frequent_ruleitems.frequent_ruleitems_set:
            new_ruleitem = join(item1, item2, dataset)
            if new_ruleitem:
                returned_frequent_ruleitems.add(new_ruleitem)
                if returned_frequent_ruleitems.get_size() >= 1000:      # not allow to store more than 1000 ruleitems
                    return returned_frequent_ruleitems
    return returned_frequent_ruleitems

def rule_generator(dataset: Dataset, minsup_dict: dict, minconf: float) -> Car:
    """
    Main method, implementation of CBA-RG algorithm
    """
    frequent_ruleitems = FrequentRuleitems()
    car = Car()

    # FIRST SCAN (C1)
    for column in range(dataset.num_attributes):
        distinct_value = dataset.get_distinct_values(column)
        for value in distinct_value:
            cond_set = {column: value}
            for classes in set(dataset.get_class_list()):
                rule_item = ruleitem.RuleItem(cond_set, classes, dataset)
                minsup = get_minsup(classes, minsup_dict)
                if rule_item.support >= minsup:
                    frequent_ruleitems.add(rule_item)

    car.gen_rules(frequent_ruleitems, minsup_dict, minconf) # pass in indiv rule items, get rule item whcih includes minsup and conf
    cars = car
    last_cars_number = 0
    current_cars_number = len(car.rules)

    # Combine and generate ruleitems while ruletimes are valid
    while frequent_ruleitems.get_size() > 0 and current_cars_number <= 2000 and (current_cars_number - last_cars_number) >= 10:
        candidate = candidate_gen(frequent_ruleitems, dataset)  # generate candidate ruletimes
        frequent_ruleitems = FrequentRuleitems()
        car = Car()
        for item in candidate.frequent_ruleitems_set:
            label = item.class_label
            minsup = get_minsup(label, minsup_dict)
            if item.support >= minsup:
                frequent_ruleitems.add(item)
        car.gen_rules(frequent_ruleitems, minsup_dict, minconf)
        cars.append(car, minsup, minconf)
        last_cars_number = current_cars_number
        current_cars_number = len(cars.rules)
    return cars


