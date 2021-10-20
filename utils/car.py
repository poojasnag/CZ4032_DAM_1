from utils.prune import *
from utils.pre_processing import *

class Car:
    """
    Classification Association Rule (CAR)
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
    def gen_rules(self, frequent_ruleitems, minsup_dict, minconf):
        for item in frequent_ruleitems.frequent_ruleitems_set:
            label = item.class_label
            minsup = get_minsup(label, minsup_dict)
            self._add(item, minsup, minconf)

    # prune rules
    def prune_rules(self, dataset):
        for rule in self.rules:
            pruner = Prune(rule, dataset.data)
            pruner.find_prune_rule(rule)
            pruned_rule = pruner.pruned_rule
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
