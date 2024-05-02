import torch

class CrossEntropyLoss:
    def __call__(self, predicted, target):
        return torch.nn.functional.cross_entropy(predicted, target)
