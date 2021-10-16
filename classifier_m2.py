class Classifier_m2:
    """
    The definition of classifier formed in CBA-CB: M2. It contains a list of rules order by their precedence, a default
    class label. The other member are private and useless for outer code.
    """
    def __init__(self):
        self.rule_list = list()
        self.default_class = None
        self._default_class_list = list()
        self._total_errors_list = list()

    # insert a new rule into classifier
    def add(self, rule, default_class, total_errors):
        self.rule_list.append(rule)
        self._default_class_list.append(default_class)
        self._total_errors_list.append(total_errors)

    # discard those rules that introduce more errors. See line 18-20, CBA-CB: M2 (Stage 3).
    def discard(self):
        index = self._total_errors_list.index(min(self._total_errors_list))
        self.rule_list = self.rule_list[:(index + 1)]
        self._total_errors_list = None

        self.default_class = self._default_class_list[index]
        self._default_class_list = None

    # just print out rules and default class label
    def print(self):
        for rule in self.rule_list:
            rule.print_rule()
        print("default_class:", self.default_class)
