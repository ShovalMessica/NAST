import torch

class DiversityLoss:
    def __call__(self, predictions):
        diversity_loss = 0
        mean_predictions = torch.sum(predictions, 0) / predictions.shape[0]
        log_probs = torch.log(mean_predictions)
        for x, y in zip(mean_predictions, log_probs):
            diversity_loss += x * y

        return diversity_loss
