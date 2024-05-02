import torch

class ReconstructionLoss:
    def __call__(self, reconstructed, target):
        return torch.nn.functional.l1_loss(reconstructed, target)
