class FrequentRuleitems:
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