from utils.ruleitem import RuleItem

class Rule(RuleItem):
    """
    A class inherited from RuleItem, adding classCasesCovered and replace field.
    """
    def __init__(self, cond_set, class_label, dataset):
        RuleItem.__init__(self, cond_set, class_label, dataset)
        self._init_classCasesCovered(dataset)
        self.replace = set()

    # initialize the classCasesCovered field
    def _init_classCasesCovered(self, dataset):
        class_column = dataset.get_class_list() # [x[-1] for x in dataset]
        class_label = set(class_column)
        self.classCasesCovered = dict((x, 0) for x in class_label)
