import torch

class DiversityLoss:
    def __call__(self, predictions, num_units):
        mean_predictions = torch.sum(predictions, 0) / predictions.shape[0]
        log_probs = torch.log(mean_predictions)
        diversity_loss = torch.sum(mean_predictions * log_probs)
        return diversity_loss / num_units
