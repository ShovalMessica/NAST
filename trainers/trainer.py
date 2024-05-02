import os
import yaml
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from utils.config import load_config
from fairseq.examples.hubert.simple_kmeans.dump_hubert_feature import HubertFeatureReader
from augmentations.audio_augmentations import augment
from losses.reconstruction_loss import ReconstructionLoss
from losses.diversity_loss import DiversityLoss
from losses.cross_entropy_loss import CrossEntropyLoss
from utils.training_utils import adjust_cross_entropy_weight, synchronize_diversity_weight
from utils.logger import get_logger
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, config_path, checkpoint_dir):
        self.model = model
        self.feature_extractor = HubertFeatureReader(HUBERT_CKPT_PATH, layer=9, max_chunk=1600000).eval()
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_config = self.load_config(config_path)
        self.reconstruction_loss = ReconstructionLoss()
        self.diversity_loss = DiversityLoss()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.logger = get_logger("Trainer")
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')
        self.checkpoint_interval = self.training_config['training']['checkpoint_interval']

    def train(self):
        num_epochs = self.training_config['training']['num_epochs']
        batch_size = self.training_config['training']['batch_size']
        validation_interval = self.training_config['training']['validation_interval']
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
                augmented_audio = [augment(x) for x in clean_audio]

                # Extract HuBERT features from the audio file
                clean_features = [self.feature_extractor.get_feats(x) for x in clean_audio]
                augmented_features = [self.feature_extractor.get_feats(x) for x in augmented_audio]

                target_features = clean_features if self.model.reconstruction_type == "HuBERT" else [extract_mfcc_features(x).to(device).requires_grad_() for x in batch]

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
            if (epoch + 1) % validation_interval == 0:
                val_loss = self.validate(val_loader, epoch)

                # Save the best model checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, batch_idx, is_best=True)

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
            if self.training_config['phase2']['losses']['cross_entropy']:
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

    def log_losses(self, loss_dict, epoch, batch_idx, num_batches):
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.logger.info(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{num_batches}], {loss_name.capitalize()}: {loss_value:.4f}")
        self.logger.info(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{num_batches}], Total Loss: {loss_dict['total_loss']:.4f}")

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss_dict = {
            'reconstruction_loss': 0.0,
            'diversity_loss': 0.0,
            'ce_loss': 0.0,
            'total_loss': 0.0
        }

        with torch.no_grad():
            for batch in val_loader:
                clean_audio = [self.feature_extractor.read_audio(x) for x in batch]
                clean_features = [self.feature_extractor.get_feats(x) for x in clean_audio]
                target_features = clean_features if self.model.reconstruction_type == "HuBERT" else [extract_mfcc_features(x).to(device).requires_grad_() for x in batch]

                for x, rec_target in zip(clean_features, target_features):
                    rec_x, one_hot_x, predicts_x = self.model(x)

                    if self.training_config['phase1']['losses']['reconstruction']:
                        val_loss_dict['reconstruction_loss'] += self.reconstruction_loss(rec_x, rec_target)
                    if self.training_config['phase1']['losses']['diversity']:
                        val_loss_dict['diversity_loss'] += self.diversity_loss(one_hot_x, self.model.num_units)
                    if self.training_config['phase2']['losses']['cross_entropy']:
                        val_loss_dict['ce_loss'] += self.cross_entropy_loss(predicts_x, one_hot_x)

        batch_size = len(val_loader.dataset)
        val_loss_dict['reconstruction_loss'] /= batch_size
        val_loss_dict['diversity_loss'] /= batch_size
        val_loss_dict['ce_loss'] /= batch_size

        if epoch < self.training_config['phase1']['epochs']:
            # Phase 1: Reconstruction and Diversity Loss
            if self.training_config['phase1']['losses']['reconstruction']:
                val_loss_dict['total_loss'] += self.training_config['phase1']['weights']['reconstruction'] * val_loss_dict['reconstruction_loss']
            if self.training_config['phase1']['losses']['diversity']:
                val_loss_dict['total_loss'] += self.training_config['phase1']['weights']['diversity'] * val_loss_dict['diversity_loss']
        else:
            # Phase 2: All Losses
            if self.training_config['phase2']['losses']['reconstruction']:
                val_loss_dict['total_loss'] += self.training_config['phase2']['weights']['reconstruction'] * val_loss_dict['reconstruction_loss']
            if self.training_config['phase2']['losses']['diversity']:
                val_loss_dict['total_loss'] += self.training_config['phase2']['weights']['diversity'] * val_loss_dict['diversity_loss']
            if self.training_config['phase2']['losses']['cross_entropy']:
                val_loss_dict['total_loss'] += self.training_config['phase2']['weights']['cross_entropy'] * val_loss_dict['ce_loss']

        self.logger.info(f"Epoch [{epoch+1}] Validation Losses:")
        for loss_name, loss_value in val_loss_dict.items():
            if loss_name != 'total_loss':
                self.logger.info(f"{loss_name.capitalize()}: {loss_value:.4f}")
        self.logger.info(f"Total Validation Loss: {val_loss_dict['total_loss']:.4f}")

        self.model.train()
        return val_loss_dict['total_loss']

    def save_checkpoint(self, epoch, batch_idx, is_best=False):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}_batch_{batch_idx}.pt")
        save_checkpoint(self.model, checkpoint_path)

        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            save_checkpoint(self.model, best_checkpoint_path)
            self.logger.info(f"Best model checkpoint saved: {best_checkpoint_path}")
        else:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
