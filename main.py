import argparse
import yaml
import torch
from torch.optim import Adam
from network import Network
from trainer import Trainer
from datasets.paths_dataset import PathsDataset
from fairseq.examples.hubert.simple_kmeans.dump_hubert_feature import HubertFeatureReader
from augmentations.audio_augmentations import AudioAugmentations

def main(args):
    # Load the configuration files
    with open(args.training_config, 'r') as f:
        training_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained HuBERT feature extractor
    hubert_ckpt_path = training_config['checkpoints']['hubert']
    feature_extractor = HubertFeatureReader(hubert_ckpt_path, layer=9)

    # Load the model configuration
    model_config = model_config[args.model_name]
    model = Network(config=model_config, device=device)

    # Load the optimizer
    optimizer = Adam(model.parameters(), lr=training_config['training']['learning_rate'])

    # Load the datasets
    train_dataset = PathsDataset(tsv_file=training_config['datasets']['train_tsv_path'])
    val_dataset = PathsDataset(tsv_file=training_config['datasets']['val_tsv_path'])

    # Create the audio augmentations
    audio_augmentations = AudioAugmentations(training_config)

    # Create the trainer
    trainer = Trainer(model, optimizer, train_dataset, val_dataset, args.training_config, args.checkpoint_dir)

    # Start training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the model.')
    parser.add_argument('--training_config', type=str, required=True, help='Path to the training configuration file.')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--model_name', type=str, required=True, choices=['50_units', '100_units', '200_units'], help='Name of the model configuration to use.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save the model checkpoints.')
    args = parser.parse_args()

    main(args)
