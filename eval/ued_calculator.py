import os
import torch
from torch.utils.data import DataLoader
from fairseq.examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
from utils.training_utils import read_audio, get_feats
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from augmentations.transformations import AudioAugmentations
from jiwer import wer
from models.network import Network
from datasets.paths_dataset import PathsDataset
from typing import List
import argparse

log_dir = os.path.join(os.path.dirname(__file__), '..', 'runs/UED')
writer = SummaryWriter(log_dir)


class UEDCalculator:
    def __init__(self, network, feature_extractor, config, device):
        self.network = network
        self.feature_extractor = feature_extractor
        self.device = device
        self.audio_augmentations = AudioAugmentations(config, phase='phase2').augmentations
        self.augmentations_str = ["Noise 5-30", "Time Stretch", "Pitch Shift", "Reverberation"]

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
                clean_audio_batch = [read_audio(self.feature_extractor, x) for x in batch]
                augmented_audio_batch = [augmentation(x) for x in clean_audio_batch]

                clean_features = [get_feats(self.feature_extractor, x) for x in clean_audio_batch]
                augmented_features = [get_feats(self.feature_extractor, x).to(self.device) for x in
                                      augmented_audio_batch]

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
        writer.add_scalar(augmentation_str, ued, int(ckpt_path.split('_')[1]),
                          int(ckpt_path.split('_')[-1].split('.')[0]))

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
        print("Starting UED scores calculation ...")
        for augmentation, augmentation_str in zip(self.audio_augmentations, self.augmentations_str):
            ued = self.calc_ued(augmentation, augmentation_str, dataset, ckpt_path)
            ueds.append(ued)

        return ueds


def main():
    parser = argparse.ArgumentParser(description="Calculate UED for different augmentations.")
    parser.add_argument("--dataset_tsv_path", type=str, required=True,
                        help="Path to the TSV file containing the dataset for UED calculation.")
    parser.add_argument("--training_config_path", type=str, required=True, help="Path to the training configuration "
                                                                                "YAML file.")
    parser.add_argument("--model_config_path", type=str, required=True,
                        help="Path to the model configuration YAML file.")
    parser.add_argument("--model_ckpt_path", type=str, required=True,
                        help="Path to the model checkpoints.")

    args = parser.parse_args()

    config = load_config(args.training_config_path, args.model_config_path)

    # Load the pre-trained model
    feature_extractor = HubertFeatureReader(config['checkpoints']['hubert'], layer=9, max_chunk=1600000)
    network = Network(config=config)
    load_checkpoint(network, args.model_ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # Load the dataset
    dataset = PathsDataset(tsv_file=args.dataset_tsv_path)

    # Create the UEDCalculator
    ued_calculator = UEDCalculator(network, feature_extractor, config, device)

    # Calculate UED for all augmentations
    ued_calculator.calc_ued_all_augmentations(dataset, args.model_ckpt_path)


if __name__ == "__main__":
    main()
    writer.close()
