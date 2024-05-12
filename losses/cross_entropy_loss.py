import torch

class CrossEntropyLoss:
    def __call__(self, predicted, target):
        # Augmentations alter the dimensions of the signal, linear interpolation is required to align and compare the
        # two signals.
        transpose = torch.unsqueeze(predicted.T, dim=0)
        predicted_interpolated = torch.nn.functional.interpolate(transpose, target.shape[0])
        predicted_interpolated = torch.squeeze(predicted_interpolated, dim=0).T
        
        return torch.nn.functional.cross_entropy(predicted_interpolated, target)
