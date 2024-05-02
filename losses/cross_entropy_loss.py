import torch

class CrossEntropyLoss:
    def __call__(self, predicted, target):
        traspose = torch.unsqueeze(predicted.T, dim=0)
        predicted_interpolated = torch.nn.functional.interpolate(traspose, target.shape[0])
        predicted_interpolated = torch.squeeze(predicted_interpolated, dim=0).T
        
        return torch.nn.functional.cross_entropy(predicted, target)
