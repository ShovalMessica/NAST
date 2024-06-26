import utils.override
from typing import Dict
import torch
from torch.utils.data import DataLoader
from datasets.paths_dataset import PathsDataset
from fairseq.examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
from augmentations.transformations import AudioAugmentations
from losses.reconstruction_loss import ReconstructionLoss
from losses.diversity_loss import DiversityLoss
from losses.cross_entropy_loss import CrossEntropyLoss
from utils.training_utils import adjust_cross_entropy_weight, synchronize_diversity_weight, read_audio, get_feats
from utils.logger import get_logger
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoint import save_checkpoint
from eval.validator import Validator


class Trainer:
    def __init__(self, model, optimizer, config, checkpoint_dir, device):
        self.model = model
        self.optimizer = optimizer
        self.training_config = config
        self.num_units = model.num_units
        self.feature_extractor = HubertFeatureReader(self.training_config['checkpoints']['hubert'], layer=9)
        self.train_dataset = PathsDataset(tsv_file=self.training_config['datasets']['train_tsv_path'])
        self.val_dataset = PathsDataset(tsv_file=self.training_config['datasets']['val_tsv_path'])
        self.reconstruction_loss = ReconstructionLoss()
        self.diversity_loss = DiversityLoss()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.logger = get_logger("Trainer")
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = torch.tensor(float('inf')).to(device)
        self.checkpoint_interval = self.training_config['training']['checkpoint_interval']
        self.validation_interval = self.training_config['training']['validation_interval']
        self.audio_augmentations = AudioAugmentations(self.training_config, phase='phase1')
        self.validator = Validator(self.model, self.feature_extractor, self.audio_augmentations, self.training_config)
        self.diversity_threshold = self.training_config["diversity"]["threshold_number"][self.num_units]
        self.device = device

    def train(self):
        writer = SummaryWriter('runs/Train')

        self.logger.info(f"Starting Training Process ...")
        num_epochs = self.training_config['training']['num_epochs']
        batch_size = self.training_config['training']['batch_size']
        log_interval = self.training_config['training']['log_interval']

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        ce_loss_weight = self.training_config['phase2']['weights']['cross_entropy']
        ce_loss_tracking = []

        diversity_weight = self.training_config['phase2']['weights']['diversity']

        for epoch in range(num_epochs):
            self.logger.info(
                f"Start training epoch [{epoch + 1}], Training Phase: {self.audio_augmentations.phase[-1]}")
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                # Read audio files
                clean_audio = [read_audio(self.feature_extractor, x) for x in batch]
                augmented_audio = [self.audio_augmentations.augment(x) for x in clean_audio]

                clean_features = [get_feats(self.feature_extractor, x) for x in clean_audio]
                augmented_features = [get_feats(self.feature_extractor, x) for x in augmented_audio]

                target_features = clean_features if self.training_config[self.num_units][
                                                        'reconstruction_type'] == "HuBERT" else None

                loss, loss_dict = self.calculate_loss(clean_features, augmented_features, target_features, epoch,
                                                      ce_loss_weight, diversity_weight)

                loss.backward()
                self.optimizer.step()

                # Log training progress
                if (batch_idx + 1) % log_interval == 0:
                    writer.add_scalar('Loss/train', loss_dict['total_loss'], epoch, batch_idx)
                    self.log_losses(loss_dict, epoch, batch_idx, len(train_loader))

                if self.audio_augmentations.phase == 'phase2':

                    ce_loss_tracking.append(loss_dict['ce_loss'])
                    if len(ce_loss_tracking) > 10:
                        ce_loss_tracking = ce_loss_tracking[-10:]

                    # Adjust Cross-Entropy Loss weight
                    ce_loss_weight = adjust_cross_entropy_weight(ce_loss_weight,
                                                                ce_loss_tracking,
                                                                self.training_config)

                    # Synchronize Diversity Loss weight
                    diversity_weight = synchronize_diversity_weight(diversity_weight,
                                                                    self.training_config,
                                                                    loss_dict['one_hot_vectors'],
                                                                    self.diversity_threshold)

                # Save checkpoint
                if (batch_idx + 1) % self.checkpoint_interval == 0:
                    save_checkpoint(self.model, epoch + 1, batch_idx + 1, self.checkpoint_dir, is_best=False)

                # Validate the model
                if (batch_idx + 1) % self.validation_interval == 0:
                    val_loss = self.validator.validate(val_loader, epoch, batch_idx)
                    writer.add_scalar('Loss/valid', val_loss, epoch, batch_idx)

                    # Save the best model checkpoint
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(self.model, epoch + 1, batch_idx + 1, self.checkpoint_dir, is_best=False)

            # Update the phase for the next epoch
            if epoch == self.training_config['phase1']['epochs'] - 1:
                self.audio_augmentations.phase = 'phase2'

        writer.close()

    def calculate_loss(self, clean_features, augmented_features, target_features, epoch, ce_loss_weight,
                       diversity_weight):
        loss_dict = {
            'reconstruction_loss': torch.tensor(0.0).to(self.device),
            'diversity_loss': torch.tensor(0.0).to(self.device),
            'ce_loss': torch.tensor(0.0).to(self.device),
            'total_loss': torch.tensor(0.0).to(self.device),
            'one_hot_vectors': []
        }

        for x, augmented_x, rec_target in zip(clean_features, augmented_features, target_features):
            rec_x, one_hot_x, predicts_x = self.model(x)
            rec_augmented_x, one_hot_augmented_x, predicts_augmented_x = self.model(augmented_x)

            loss_dict['one_hot_vectors'].append(one_hot_x)

            if self.training_config['phase1']['losses']['reconstruction']:
                loss_dict['reconstruction_loss'] += self.reconstruction_loss(rec_x, rec_target)
            if self.training_config['phase1']['losses']['diversity']:
                loss_dict['diversity_loss'] += self.diversity_loss(one_hot_x)
            if epoch >= self.training_config['phase1']['epochs'] and self.training_config['phase2']['losses'][
                'cross_entropy']:
                loss_dict['ce_loss'] += self.cross_entropy_loss(predicts_augmented_x, one_hot_x)

        batch_size = len(clean_features)
        loss_dict['reconstruction_loss'] /= batch_size
        loss_dict['diversity_loss'] /= self.num_units
        loss_dict['ce_loss'] /= batch_size

        if epoch < self.training_config['phase1']['epochs']:
            # Phase 1: Reconstruction and Diversity Loss
            if self.training_config['phase1']['losses']['reconstruction']:
                loss_dict['total_loss'] += self.training_config['phase1']['weights']['reconstruction'] * loss_dict[
                    'reconstruction_loss']
            if self.training_config['phase1']['losses']['diversity']:
                loss_dict['total_loss'] += self.training_config['phase1']['weights']['diversity'] * loss_dict[
                    'diversity_loss']
        else:
            # Phase 2: All Losses
            if self.training_config['phase2']['losses']['reconstruction']:
                loss_dict['total_loss'] += self.training_config['phase2']['weights']['reconstruction'] * loss_dict[
                    'reconstruction_loss']
            if self.training_config['phase2']['losses']['diversity']:
                loss_dict['total_loss'] += diversity_weight * loss_dict['diversity_loss']
            if self.training_config['phase2']['losses']['cross_entropy']:
                loss_dict['total_loss'] += ce_loss_weight * loss_dict['ce_loss']

        return loss_dict['total_loss'], loss_dict

    def log_losses(self, loss_dict: Dict[str, torch.Tensor], epoch: int, batch_idx: int, num_batches: int):
        reconstruction_loss = loss_dict['reconstruction_loss']
        diversity_loss = loss_dict['diversity_loss']
        ce_loss = loss_dict['ce_loss']
        total_loss = loss_dict['total_loss']

        progress = (batch_idx + 1) / num_batches
        progress_bar_length = 20
        filled_length = int(progress_bar_length * progress)
        progress_bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
        progress_percentage = f"{progress * 100:.1f}%"

        if self.audio_augmentations.phase == 'phase1':
            log_message = (
                f"Epoch [{epoch + 1}] | "
                f"[{batch_idx + 1}/{num_batches}] | "
                f"Reconstruction Loss: {reconstruction_loss:.4f}, "
                f"Diversity Loss: {diversity_loss:.4f}, "
                f"Total Loss: {total_loss:.4f}   |{progress_bar}| {progress_percentage} "
            )
        else:
            log_message = (
                f"Epoch [{epoch + 1}] | "
                f"[{batch_idx + 1}/{num_batches}] | "
                f"Reconstruction Loss: {reconstruction_loss:.4f}, "
                f"Diversity Loss: {diversity_loss:.4f}, "
                f"Cross-Entropy Loss: {ce_loss:.4f}, "
                f"Total Loss: {total_loss:.4f}   |{progress_bar}| {progress_percentage} "
            )

        print(f"\r{log_message}", end="", flush=True)

        if batch_idx + 1 == num_batches:
            print()
