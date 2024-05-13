import argparse

import torch
from models.network import Network
from trainers.trainer import Trainer
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train configuration arguments")
    parser.add_argument("--training_config_path", type=str, required=True,
                        help="Path to the training configuration YAML file.")
    parser.add_argument("--model_config_path", type=str, required=True,
                        help="Path to the model configuration YAML file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    config = load_config(args.training_config_path, args.model_config_path)

    model = Network(config=config, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Create an instance of the Trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      config=config,
                      checkpoint_dir="checkpoints/",
                      device=device)

    # Start the training process
    trainer.train()


if __name__ == "__main__":
    main()
