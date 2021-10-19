import ruleitem
from cba_cb_m2 import *
import sys

class Prune:
    def __init__(self, initial_rule, dataset):
        self.dataset = dataset
        self.min_rule_error = sys.maxsize
        self.pruned_rule = initial_rule
        
  # prune rule recursively
    def find_prune_rule(self, this_rule):
        # calculate how many errors the rule r make in the dataset
        rule_error = errorsOfRule(this_rule, self.dataset)
        if rule_error < self.min_rule_error:
            self.min_rule_error = rule_error
            self.pruned_rule = this_rule
        this_rule_cond_set = list(this_rule.cond_set)
        if len(this_rule_cond_set) >= 2:
            for attribute in this_rule_cond_set:
                temp_cond_set = dict(this_rule.cond_set)
                temp_cond_set.pop(attribute)
                temp_rule = ruleitem.RuleItem(temp_cond_set, this_rule.class_label, self.dataset)
                temp_rule_error = errorsOfRule(temp_rule, self.dataset)
                if temp_rule_error <= self.min_rule_error:
                    self.min_rule_error = temp_rule_error
                    self.pruned_rule = temp_rule
                    if len(temp_cond_set) >= 2:
                        self.find_prune_rule(temp_rule)