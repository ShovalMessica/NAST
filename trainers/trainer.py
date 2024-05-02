import torch
from torch.utils.data import DataLoader
from models.hubert_network import HubertNetwork
from utils.data_utils import extract_hubert_features
from augmentations.audio_augmentations import augment
from losses.reconstruction_loss import ReconstructionLoss
from losses.diversity_loss import DiversityLoss
from losses.cross_entropy_loss import CrossEntropyLoss
from utils.training_utils import adjust_cross_entropy_weight, synchronize_diversity_weight
from utils.logger import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HubertTrainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, training_config):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_config = training_config
        self.reconstruction_loss = ReconstructionLoss()
        self.diversity_loss = DiversityLoss()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.logger = get_logger("HubertTrainer")
    
    def train(self):
        num_epochs = self.training_config['training']['num_epochs']
        batch_size = self.training_config['training']['batch_size']
        checkpoint_interval = self.training_config['training']['checkpoint_interval']
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
                
                clean_audio = [extract_hubert_features(x) for x in batch]
                augmented_audio = [augment(x) for x in clean_audio]
                
                clean_features = [extract_hubert_features(x) for x in clean_audio]
                augmented_features = [extract_hubert_features(x).to(device) for x in augmented_audio]
                
                target_features = clean_features if self.model.reconstruction_type == "HuBERT"
                
                loss = 0.0
                reconstruction_loss_value = 0.0
                diversity_loss_value = 0.0
                ce_loss_value = 0.0
                
                for x, augmented_x, rec_target in zip(clean_features, augmented_features, target_features):
                    rec_x, one_hot_x, predicts_x = self.model(x)
                    rec_augmented_x, one_hot_augmented_x, predicts_augmented_x = self.model(augmented_x)
                    
                    if self.training_config['phase1']['losses']['reconstruction']:
                        reconstruction_loss_value += self.reconstruction_loss(rec_x, rec_target)
                    if self.training_config['phase1']['losses']['diversity']:
                        diversity_loss_value += self.diversity_loss(one_hot_x, self.model.num_units)
                    if self.training_config['phase2']['losses']['cross_entropy']:
                        ce_loss_value += self.cross_entropy_loss(predicts_augmented_x, one_hot_x)
                
                reconstruction_loss_value /= len(batch)
                diversity_loss_value /= len(batch)
                ce_loss_value /= len(batch)
                
                if epoch < self.training_config['phase1']['epochs']:
                    # Phase 1: Reconstruction and Diversity Loss
                    if self.training_config['phase1']['losses']['reconstruction']:
                        loss += self.training_config['phase1']['weights']['reconstruction'] * reconstruction_loss_value
                    if self.training_config['phase1']['losses']['diversity']:
                        loss += self.training_config['phase1']['weights']['diversity'] * diversity_loss_value
                else:
                    # Phase 2: All Losses
                    if self.training_config['phase2']['losses']['reconstruction']:
                        loss += self.training_config['phase2']['weights']['reconstruction'] * reconstruction_loss_value
                    if self.training_config['phase2']['losses']['diversity']:
                        loss += diversity_weight * diversity_loss_value
                    if self.training_config['phase2']['losses']['cross_entropy']:
                        loss += ce_loss_weight * ce_loss_value
                    
                    # Adjust Cross-Entropy Loss weight
                    ce_loss_weight, ce_loss_prev, ce_loss_stabilized = adjust_cross_entropy_weight(ce_loss_value, ce_loss_prev, ce_loss_stabilized, self.training_config)
                    
                    # Synchronize Diversity Loss weight
                    diversity_weight, diversity_prev = synchronize_diversity_weight(diversity_loss_value, diversity_prev, diversity_weight, self.training_config)
                
                loss.backward()
                self.optimizer.step()
                
                # Log training progress
                if (batch_idx + 1) % log_interval == 0:
                    self.logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            # Validate the model
            if (epoch + 1) % validation_interval == 0:
                self.validate(val_loader)
            
            # Save model checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                clean_audio = [extract_hubert_features(x) for x in batch]
                clean_features = [extract_hubert_features(x) for x in clean_audio]
                target_features = clean_features if self.model.reconstruction_type == "HuBERT" else [extract_mfcc_features(x).to(device).requires_grad_() for x in batch]
                
                for x, rec_target in zip(clean_features, target_features):
                    rec_x, _, _ = self.model(x)
                    val_loss += self.reconstruction_loss(rec_x, rec_target)
        
        val_loss /= len(val_loader.dataset)
        self.logger.info(f"Validation Loss: {val_loss:.4f}")
    
    def save_checkpoint(self, epoch):
        checkpoint_path = f"checkpoints/model_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
