import os
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from datasets.paths_dataset import PathsDataset
from fairseq.examples.hubert.simple_kmeans.dump_hubert_feature import HubertFeatureReader
from augmentations.audio_augmentations import AudioAugmentations
from losses.reconstruction_loss import ReconstructionLoss
from losses.diversity_loss import DiversityLoss
from losses.cross_entropy_loss import CrossEntropyLoss
from utils.training_utils import adjust_cross_entropy_weight, synchronize_diversity_weight
from utils.logger import get_logger
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from eval.validator import Validator

class Trainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, config_path, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.training_config = load_config(config_path)
        self.feature_extractor = HubertFeatureReader(self.training_config['checkpoints']['hubert'], layer=9, max_chunk=1600000).eval()
        self.train_dataset = PathsDataset(tsv_file=self.training_config['datasets']['train_tsv_path'])
        self.val_dataset = PathsDataset(tsv_file=self.training_config['datasets']['val_tsv_path'])
        self.reconstruction_loss = ReconstructionLoss()
        self.diversity_loss = DiversityLoss()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.logger = get_logger("Trainer")
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')
        self.checkpoint_interval = self.training_config['training']['checkpoint_interval']
        self.validation_interval = self.training_config['training']['validation_interval']
        self.audio_augmentations = AudioAugmentations(self.training_config, phase='phase1')
        self.validator = Validator(self.model, self.feature_extractor, self.audio_augmentations, self.training_config)

    def train(self):
        num_epochs = self.training_config['training']['num_epochs']
        batch_size = self.training_config['training']['batch_size']
        log_interval = self.training_config['training']['log_interval']

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        ce_loss_weight = self.training_config['phase2']['weights']['cross_entropy']
        ce_loss_stabilized = False
        ce_loss_prev = float('inf')

        diversity_weight = self.training_config['phase2']['weights']['diversity']
        diversity_prev = 0.0

        for epoch in range(num_epochs):
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                # Read audio files
                clean_audio = [self.feature_extractor.read_audio(x) for x in batch]
                augmented_audio = [self.audio_augmentations.augment(x) for x in clean_audio]

                # Extract HuBERT features from the audio file
                clean_features = [self.feature_extractor.get_feats(x) for x in clean_audio]
                augmented_features = [self.feature_extractor.get_feats(x) for x in augmented_audio]

                target_features = clean_features if self.model.reconstruction_type == "HuBERT"

                loss, loss_dict = self.calculate_loss(clean_features, augmented_features, target_features, epoch, ce_loss_weight, diversity_weight)

                loss.backward()
                self.optimizer.step()

                # Log training progress
                if (batch_idx + 1) % log_interval == 0:
                    self.log_losses(loss_dict, epoch, batch_idx, len(train_loader))

                # Adjust Cross-Entropy Loss weight
                ce_loss_weight, ce_loss_prev, ce_loss_stabilized = adjust_cross_entropy_weight(loss_dict['ce_loss'], ce_loss_prev, ce_loss_stabilized, self.training_config)

                # Synchronize Diversity Loss weight
                diversity_weight, diversity_prev = synchronize_diversity_weight(loss_dict['diversity_loss'], diversity_prev, diversity_weight, self.training_config)

                # Save checkpoint
                if (batch_idx + 1) % self.checkpoint_interval == 0:
                    self.save_checkpoint(epoch, batch_idx, is_best=False)

                # Validate the model
                if (batch_idx + 1) % self.validation_interval == 0:
                    val_loss = self.validator.validate(val_loader, epoch, batch_idx)

                    # Save the best model checkpoint
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, batch_idx, is_best=True)

            # Update the phase for the next epoch
            if epoch == self.training_config['phase1']['epochs'] - 1:
                self.audio_augmentations.phase = 'phase2'

    def calculate_loss(self, clean_features, augmented_features, target_features, epoch, ce_loss_weight, diversity_weight):
        loss_dict = {
            'reconstruction_loss': 0.0,
            'diversity_loss': 0.0,
            'ce_loss': 0.0,
            'total_loss': 0.0
        }

        for x, augmented_x, rec_target in zip(clean_features, augmented_features, target_features):
            rec_x, one_hot_x, predicts_x = self.model(x)
            rec_augmented_x, one_hot_augmented_x, predicts_augmented_x = self.model(augmented_x)

            if self.training_config['phase1']['losses']['reconstruction']:
                loss_dict['reconstruction_loss'] += self.reconstruction_loss(rec_x, rec_target)
            if self.training_config['phase1']['losses']['diversity']:
                loss_dict['diversity_loss'] += self.diversity_loss(one_hot_x, self.model.num_units)
            if epoch >= self.training_config['phase1']['epochs']: and self.training_config['phase2']['losses']['cross_entropy']:
                loss_dict['ce_loss'] += self.cross_entropy_loss(predicts_augmented_x, one_hot_x)

        batch_size = len(clean_features)
        loss_dict['reconstruction_loss'] /= batch_size
        loss_dict['diversity_loss'] /= batch_size
        loss_dict['ce_loss'] /= batch_size

        if epoch < self.training_config['phase1']['epochs']:
            # Phase 1: Reconstruction and Diversity Loss
            if self.training_config['phase1']['losses']['reconstruction']:
                loss_dict['total_loss'] += self.training_config['phase1']['weights']['reconstruction'] * loss_dict['reconstruction_loss']
            if self.training_config['phase1']['losses']['diversity']:
                loss_dict['total_loss'] += self.training_config['phase1']['weights']['diversity'] * loss_dict['diversity_loss']
        else:
            # Phase 2: All Losses
            if self.training_config['phase2']['losses']['reconstruction']:
                loss_dict['total_loss'] += self.training_config['phase2']['weights']['reconstruction'] * loss_dict['reconstruction_loss']
            if self.training_config['phase2']['losses']['diversity']:
                loss_dict['total_loss'] += diversity_weight * loss_dict['diversity_loss']
            if self.training_config['phase2']['losses']['cross_entropy']:
                loss_dict['total_loss'] += ce_loss_weight * loss_dict['ce_loss']

        return loss_dict['total_loss'], loss_dict

    def log_losses(self, loss_dict: Dict[str, float], epoch: int, batch_idx: int, num_batches: int):
        reconstruction_loss = loss_dict['reconstruction_loss']
        diversity_loss = loss_dict['diversity_loss']
        ce_loss = loss_dict['ce_loss']
        total_loss = loss_dict['total_loss']

        self.logger.info(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{num_batches}], Reconstruction Loss: {reconstruction_loss:.4f}")
        self.logger.info(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{num_batches}], Diversity Loss: {diversity_loss:.4f}")
        self.logger.info(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{num_batches}], Cross-Entropy Loss: {ce_loss:.4f}")
        self.logger.info(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{num_batches}], Total Loss: {total_loss:.
