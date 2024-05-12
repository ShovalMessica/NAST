import torch
from typing import List, Dict
from losses.reconstruction_loss import ReconstructionLoss
from losses.diversity_loss import DiversityLoss
from losses.cross_entropy_loss import CrossEntropyLoss
from utils.training_utils import read_audio, get_feats
from utils.logger import get_logger


class Validator:
    def __init__(self, model, feature_extractor, audio_augmentations, training_config):
        self.model = model
        self.feature_extractor = feature_extractor
        self.audio_augmentations = audio_augmentations
        self.training_config = training_config
        self.reconstruction_loss = ReconstructionLoss()
        self.diversity_loss = DiversityLoss()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.logger = get_logger("Validator")

    def calculate_validation_losses(self, clean_features, augmented_features, target_features, epoch):
        val_loss_dict = {
            'reconstruction_loss': 0.0,
            'diversity_loss': 0.0,
            'ce_loss': 0.0,
            'total_loss': 0.0
        }

        for x, augmented_x, rec_target in zip(clean_features, augmented_features, target_features):
            rec_x, one_hot_x, predicts_x = self.model(x)
            rec_augmented_x, one_hot_augmented_x, predicts_augmented_x = self.model(augmented_x)

            if self.training_config['phase1']['losses']['reconstruction']:
                val_loss_dict['reconstruction_loss'] += self.reconstruction_loss(rec_x, rec_target)
            if self.training_config['phase1']['losses']['diversity']:
                val_loss_dict['diversity_loss'] += self.diversity_loss(one_hot_x)
            if self.training_config['phase2']['losses']['cross_entropy']:
                val_loss_dict['ce_loss'] += self.cross_entropy_loss(predicts_augmented_x, one_hot_x)

        batch_size = len(clean_features)
        val_loss_dict['reconstruction_loss'] /= batch_size
        val_loss_dict['diversity_loss'] /= self.model.num_units
        val_loss_dict['ce_loss'] /= batch_size

        return val_loss_dict

    def calculate_total_validation_loss(self, val_loss_dict, epoch):
        total_loss = 0.0

        if epoch < self.training_config['phase1']['epochs']:
            # Phase 1: Reconstruction and Diversity Loss
            if self.training_config['phase1']['losses']['reconstruction']:
                total_loss += self.training_config['phase1']['weights']['reconstruction'] * val_loss_dict[
                    'reconstruction_loss']
            if self.training_config['phase1']['losses']['diversity']:
                total_loss += self.training_config['phase1']['weights']['diversity'] * val_loss_dict['diversity_loss']
        else:
            # Phase 2: All Losses
            if self.training_config['phase2']['losses']['reconstruction']:
                total_loss += self.training_config['phase2']['weights']['reconstruction'] * val_loss_dict[
                    'reconstruction_loss']
            if self.training_config['phase2']['losses']['diversity']:
                total_loss += self.training_config['phase2']['weights']['diversity'] * val_loss_dict['diversity_loss']
            if self.training_config['phase2']['losses']['cross_entropy']:
                total_loss += self.training_config['phase2']['weights']['cross_entropy'] * val_loss_dict['ce_loss']

        return total_loss

    def validate(self, val_loader, epoch, batch_idx) -> float:
        self.logger.info(f"Starting validation for Epoch [{epoch + 1}], Batch [{batch_idx + 1}]")
        self.model.eval()

        val_loss_dicts = []

        with torch.no_grad():
            for batch in val_loader:
                clean_audio = [read_audio(self.feature_extractor, x) for x in batch]
                augmented_audio = [self.audio_augmentations.augment(x) for x in clean_audio]
                clean_features = [get_feats(self.feature_extractor, x) for x in clean_audio]
                augmented_features = [get_feats(self.feature_extractor, x) for x in augmented_audio]
                target_features = clean_features if self.training_config[self.model.num_units]['reconstruction_type'] == "HuBERT" else None

                val_loss_dict = self.calculate_validation_losses(clean_features, augmented_features, target_features,
                                                                 epoch)
                val_loss_dicts.append(val_loss_dict)

        avg_val_loss_dict = self.average_loss_dicts(val_loss_dicts)
        total_val_loss = self.calculate_total_validation_loss(avg_val_loss_dict, epoch)

        self.log_validation_losses(avg_val_loss_dict, epoch, batch_idx)
        self.logger.info(f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Total Validation Loss: {total_val_loss:.4f}")

        self.model.train()
        return total_val_loss

    def average_loss_dicts(self, loss_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        avg_loss_dict = {key: sum(d[key] for d in loss_dicts) / len(loss_dicts) for key in loss_dicts[0]}
        return avg_loss_dict

    def log_validation_losses(self, val_loss_dict: Dict[str, float], epoch: int, batch_idx: int):
        self.logger.info(f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Validation Losses:")
        self.logger.info(f"Reconstruction Loss: {val_loss_dict['reconstruction_loss']:.4f}")
        self.logger.info(f"Diversity Loss: {val_loss_dict['diversity_loss']:.4f}")
        self.logger.info(f"Cross-Entropy Loss: {val_loss_dict['ce_loss']:.4f}")
