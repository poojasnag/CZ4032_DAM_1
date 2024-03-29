class RuleItem:
    """
    Ruleitem class with attributes such as condset, class_label, cond_sup_count etc. Methods to claculate consup, support, conf.
    """
    def __init__(self, cond_set, class_label, dataset):
        self.cond_set = cond_set
        self.class_label = class_label
        self.cond_sup_count, self.rule_sup_count = self._get_sup_count(dataset)
        self.support = self._get_support(len(dataset))
        self.confidence = self._get_confidence()

    # calculate condsupCount and rulesupCount
    def _get_sup_count(self, dataset):
        cond_sup_count = 0
        rule_sup_count = 0
        for case in dataset:
            is_contained = True
            for index in self.cond_set:
                if self.cond_set[index] != case[index]:
                    is_contained = False
                    break
            if is_contained:
                cond_sup_count += 1
                if self.class_label == case[-1]:
                    rule_sup_count += 1
        return cond_sup_count, rule_sup_count

    # calculate support count
    def _get_support(self, dataset_size):
        return self.rule_sup_count / dataset_size

    # calculate confidence
    def _get_confidence(self):
        if self.cond_sup_count != 0:
            return self.rule_sup_count / self.cond_sup_count
        else:
            return 0

    # print out the ruleitem
    def print(self):
        cond_set_output = ''
        for item in self.cond_set:
            cond_set_output += '(' + str(item) + ', ' + str(self.cond_set[item]) + '), '
        cond_set_output = cond_set_output[:-2]
        print('<({' + cond_set_output + '}, ' + str(self.cond_sup_count) + '), (' +
              '(class, ' + str(self.class_label) + '), ' + str(self.rule_sup_count) + ')>')

    # print out rule
    def print_rule(self):
        cond_set_output = ''
        for item in self.cond_set:
            cond_set_output += '(' + str(item) + ', ' + str(self.cond_set[item]) + '), '
        cond_set_output = '{' + cond_set_output[:-2] + '}'
        print(cond_set_output + ' -> (class, ' + str(self.class_label) + ')')

