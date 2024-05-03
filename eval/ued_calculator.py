import torch
from torch.utils.data import DataLoader
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
from augmentations.metrics_transformations import trans_noise_0_5, trans_noise_5_30, trans_time_stretch, trans_pitch_shift, trans_reverberation
from jiwer import wer
from datasets.paths_dataset import PathsDataset
import numpy as np
from typing import List
from utils.checkpoint_utils import load_checkpoint
import wandb
import argparse
import yaml

class UEDCalculator:
    def __init__(self, network, feature_extractor, device, wandb_project=None, wandb_entity=None):
        self.network = network
        self.feature_extractor = feature_extractor
        self.device = device
        self.augmentations = [trans_pitch_shift, trans_reverberation, trans_noise_5_30, trans_time_stretch]
        self.augmentations_str = ["Pitch Shift", "Reverberation", "Noise 5-30", "Time Stretch"]
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

    def deduped(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deduplicate a tensor by removing consecutive duplicates.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Deduplicated tensor.
        """
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        mask[1:] = x[1:] != x[:-1]
        return x[mask]

    def calc_ued(self, augmentation, augmentation_str: str, dataset: PathsDataset, ckpt_path: str = None) -> float:
        """
        Calculate the Unit Edit Distance (UED) for a specific augmentation.

        Args:
            augmentation: Augmentation function.
            augmentation_str (str): Name of the augmentation.
            dataset (PathsDataset): Dataset to calculate UED on.
            ckpt_path (str, optional): Path to the checkpoint file. If provided, the model weights will be loaded from the checkpoint.

        Returns:
            float: UED for the augmentation.
        """
        if ckpt_path is not None:
            load_checkpoint(self.network, ckpt_path)

        self.network.eval()
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        total_ued = 0.0
        total_units = 0

        with torch.no_grad():
            for batch in data_loader:
                clean_audio_batch = [self.feature_extractor.read_audio(x) for x in batch]
                augmented_audio_batch = [augmentation(x) for x in clean_audio_batch]

                clean_features = [self.feature_extractor.get_feats(x) for x in clean_audio_batch]
                augmented_features = [self.feature_extractor.get_feats(x).to(self.device) for x in augmented_audio_batch]

                target_labels, predicted_labels = [], []
                for clean_feat, augmented_feat in zip(clean_features, augmented_features):
                    _, clean_labels, _ = self.network(clean_feat)
                    _, augmented_labels, _ = self.network(augmented_feat)

                    target_labels.append(self.deduped(torch.argmax(clean_labels, dim=1)))
                    predicted_labels.append(self.deduped(torch.argmax(augmented_labels, dim=1)))

                for target, predicted in zip(target_labels, predicted_labels):
                    target_str = ' '.join(str(x) for x in target.tolist())
                    predicted_str = ' '.join(str(x) for x in predicted.tolist())

                    ued = wer(target_str, predicted_str)
                    total_ued += ued * len(target)
                    total_units += len(target)

        ued = (total_ued / total_units) * 100
        print(f"{augmentation_str} UED: {ued:.2f}%")

        if self.wandb_project is not None and self.wandb_entity is not None:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)
            wandb.log({f"{augmentation_str} UED": ued})
            wandb.finish()

        return ued

    def calc_ued_all_augmentations(self, dataset: PathsDataset, ckpt_path: str = None) -> List[float]:
        """
        Calculate the Unit Edit Distance (UED) for all augmentations.

        Args:
            dataset (PathsDataset): Dataset to calculate UED on.
            ckpt_path (str, optional): Path to the checkpoint file. If provided, the model weights will be loaded from the checkpoint.

        Returns:
            List[float]: List of UED values for each augmentation.
        """
        ueds = []
        for augmentation, augmentation_str in zip(self.augmentations, self.augmentations_str):
            ued = self.calc_ued(augmentation, augmentation_str, dataset, ckpt_path)
            ueds.append(ued)

        if self.wandb_project is not None and self.wandb_entity is not None:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)
            wandb.log({"UED": wandb.Table(data=list(zip(self.augmentations_str, ueds)), columns=["Augmentation", "UED"])})
            wandb.finish()

        return ueds

def main():
    parser = argparse.ArgumentParser(description="Calculate UED for different augmentations.")
    parser.add_argument("--dataset_tsv_path", type=str, required=True, help="Path to the TSV file containing the dataset for UED calculation.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to load the model weights from.")
    parser.add_argument("--hubert_ckpt_path", type=str, default="/path/to/hubert_base_ls960.pt", help="Path to the HuBERT checkpoint file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights and Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights and Biases entity name.")
    args = parser.parse_args()

    # Load the configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load the pre-trained model
    feature_extractor = HubertFeatureReader(args.hubert_ckpt_path, layer=9, max_chunk=1600000)
    network = Network(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # Load the dataset
    dataset = PathsDataset(tsv_file=args.dataset_tsv_path)

    # Create the UEDCalculator
    ued_calculator = UEDCalculator(network, feature_extractor, device, args.wandb_project, args.wandb_entity)

    # Calculate UED for all augmentations
    ued_calculator.calc_ued_all_augmentations(dataset, args.checkpoint_path)

if __name__ == "__main__":
    main()
