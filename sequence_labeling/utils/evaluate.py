from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from seqeval.scheme import IOB2


class Evaluator:
    def __init__(self):
        super(Evaluator, self).__init__()

    def acc(self, y_true, y_pred):
        y_pred = self.trans_pad_to_U(y_pred)
        acc_value = accuracy_score([y_true], [y_pred])
        return acc_value

    def precision(self, y_true, y_pred):
        y_pred = self.trans_pad_to_U(y_pred)
        precision_value = precision_score([y_true], [y_pred], mode='strict', scheme=IOB2)
        return precision_value

    def recall(self, y_true, y_pred):
        y_pred = self.trans_pad_to_U(y_pred)
        recall_value = recall_score([y_true], [y_pred], mode='strict', scheme=IOB2)
        return recall_value

    def f1score(self, y_true, y_pred):
        y_pred = self.trans_pad_to_U(y_pred)
        f1score_value = f1_score([y_true], [y_pred], mode='strict', scheme=IOB2)
        return f1score_value

    def allevalscore(self, y_true, y_pred):
        y_pred = self.trans_pad_to_U(y_pred)
        acc_value = accuracy_score([y_true], [y_pred])
        precision_value = precision_score([y_true], [y_pred], mode='strict', scheme=IOB2)
        recall_value = recall_score([y_true], [y_pred], mode='strict', scheme=IOB2)
        f1score_value = f1_score([y_true], [y_pred], mode='strict', scheme=IOB2)
        return acc_value, precision_value, recall_value, f1score_value

    def classifyreport(self, y_true, y_pred):
        y_pred = self.trans_pad_to_U(y_pred)
        report = classification_report([y_true], [y_pred], mode='strict', scheme=IOB2)
        return report

    # 将列表中的'[PAD]'转换为'O'，以满足strict mode的评价要求
    def trans_pad_to_U(self, alist):
        return ['O' if i in ['[PAD]', '[CLS]', '[SEP]', 'SOS', 'EOS', 'UNK'] else i for i in alist]
