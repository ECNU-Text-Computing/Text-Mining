from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class Evaluator:
    def __init__(self):
        super(Evaluator, self).__init__()

    def acc(self, y_true, y_pred):
        acc_value = accuracy_score(y_true, y_pred)
        return acc_value

    def precision(self, y_true, y_pred):
        precision_value = precision_score(y_true, y_pred)
        return precision_value

    def recall(self, y_true, y_pred):
        recall_value = recall_score(y_true, y_pred)
        return recall_value

    def f1score(self, y_true, y_pred):
        f1score_value = f1_score(y_true, y_pred)
        return f1score_value

    def allevalscore(self, y_true, y_pred):
        acc_value = accuracy_score(y_true, y_pred)
        precision_value = precision_score(y_true, y_pred)
        recall_value = recall_score(y_true, y_pred)
        f1score_value = f1_score(y_true, y_pred)
        return acc_value, precision_value, recall_value, f1score_value

    def classifyreport(self, y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report
