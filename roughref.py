# main method, implementation of CBA-RG algorithm
def rule_generator(dataset, minsup, minconf):
    frequent_ruleitems = FrequentRuleitems()
    car = Car()
    minsup_class = class_minsup(dataset, minsup)
    minsup_min = min_minsup(dataset, minsup)

    # get large 1-ruleitems and generate rules
    class_label = set([x[-2] for x in dataset]) #added in -1 to -2 
    for column in range(0, len(dataset[0])-1):  # range(4) each col is a feature
        distinct_value = set([x[column] for x in dataset])  # {1,2,3}
        for value in distinct_value:
            cond_set = {column: value}
            for classes in class_label:
                print("class", classes)
                rule_item = ruleitem.RuleItem(cond_set, classes, dataset)
                minsup = minsup_class.get(classes)
                print("minsup", minsup)
                if rule_item.support >= minsup:
                    frequent_ruleitems.add(rule_item)
                    frequent_ruleitems.frequent_ruleitems_set._add(rule_item, minsup, minconf) # added in check if can
    
    # car.gen_rules(frequent_ruleitems, minsup, minconf)
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
            if item.support >= minsup_min:
                frequent_ruleitems.add(item)
        car.gen_rules(frequent_ruleitems, minsup_min, minconf)
        cars.append(car, minsup_min, minconf)
        last_cars_number = current_cars_number
        current_cars_number = len(cars.rules)

    # print(cars.rules.pop().__dict__)
    # {'cond_set': {1: 2, 2: 1, 3: 1}, 'class_label': 'Iris-setosa', 'cond_sup_count': 18, 'rule_sup_count': 18, 'support': 0.13333333333333333, 'confidence': 1.0}

    return cars

inside car class
    def gen_rules(self, frequent_ruleitems, total_minsup, minconf, dataset):
        for item in frequent_ruleitems.frequent_ruleitems_set:
            label = item.class_label
            minsup = get_minsup(label, dataset, total_minsup)
            self._add(item, minsup, minconf)

    def generate_rules(self, frequent_ruleitems, minsup, minconf):
        add_item = frequent_ruleitems.frequent_ruleitems_set
        self._add(add_item, minsup, minconf)

inside frequentruleitems class 
            def __init__(self):
        self.frequent_ruleitems_set = set()