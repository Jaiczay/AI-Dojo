import torch


# x = Predictions y = Labels
class Metrics:
    def __init__(self, classes=10):
        self.classes = classes
        self.cm = torch.zeros((classes, classes))

    def reset(self):
        self.cm = torch.zeros((self.classes, self.classes))

    def add_batch(self, preds, lbls):
        preds = torch.argmax(torch.softmax(preds, 1), 1)
        for i in range(len(lbls)):
            self.cm[lbls[i]][preds[i]] += 1

    def get_scores(self):
        per_class_result = torch.diagonal(self.cm) / torch.sum(self.cm, 1)
        result = torch.sum(torch.diagonal(self.cm)) / torch.sum(self.cm)
        return per_class_result, result

    def get_normalized_cm(self):
        norm_cm = torch.clone(self.cm)
        for i in range(self.classes):
            norm_cm[i] = norm_cm[i] / torch.sum(norm_cm[i])
        return norm_cm
