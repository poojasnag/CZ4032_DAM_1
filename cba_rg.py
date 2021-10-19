import ruleitem
from pre_processing import *
from frequentRuleItems import *
from car import *

# invoked by candidate_gen, join two items to generate candidate
def join(item1, item2, dataset):
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
def rule_generator(dataset, minsup_dict, minconf):
    frequent_ruleitems = FrequentRuleitems()
    car = Car()
    # FIRST SCAN (C1)
    for column in range(dataset.num_attributes):
        # distinct_value = set([x[column] for x in dataset])  # {1,2,3}
        distinct_value = dataset.get_distinct_values(column)
        for value in distinct_value:
            cond_set = {column: value}
            # for classes in class_label:
            for classes in set(dataset.get_class_list()):
                rule_item = ruleitem.RuleItem(cond_set, classes, dataset)  # dataset.data
                minsup = get_minsup(classes, minsup_dict)
                if rule_item.support >= minsup:
                    frequent_ruleitems.add(rule_item)           # for indiv rule items 
    # L1
    car.gen_rules(frequent_ruleitems, minsup_dict, minconf) # pass in indiv rule items, get rule item whcih includes minsup and conf
    cars = car
    # {'cond_set': {1: 1}, 'class_label': 'Iris-versicolor', 'cond_sup_count': 72, 'rule_sup_count': 36, 'support': 0.26666666666666666, 'confidence': 0.5}
    last_cars_number = 0
    current_cars_number = len(car.rules)
    while frequent_ruleitems.get_size() > 0 and current_cars_number <= 2000 and (current_cars_number - last_cars_number) >= 10:
        candidate = candidate_gen(frequent_ruleitems, dataset)
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
    # {'cond_set': {1: 2, 2: 1, 3: 1}, 'class_label': 'Iris-setosa', 'cond_sup_count': 18, 'rule_sup_count': 18, 'support': 0.13333333333333333, 'confidence': 1.0}


# # just for test
# if __name__ == "__main__":
#     from CrossValM2 import CrossValidationM2

#     minsup = 0.01
#     minconf = 0.5
#     test_data_path = 'datasets/tic-tac-toe.data'
#     test_scheme_path = 'datasets/tic-tac-toe.names'

#     validation = CrossValidationM2(test_data_path, test_scheme_path, minsup, minconf)

#     validation.cross_validation(multiple=False, dev=True) # multiple minsups