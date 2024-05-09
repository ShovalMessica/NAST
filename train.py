import torch
from models.network import Network
from trainers.trainer import Trainer
from utils.config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training configuration
config_path = "configs/training_config.yaml"
training_config = load_config(config_path)

# Load the model configuration
model_config_path = "configs/model_config.yaml"
model_config = load_config(model_config_path)

# Create an instance of the Network
model = Network(config=model_config, device=device)

# Create an optimizer for the model
optimizer = torch.optim.Adam(model.parameters(), lr=training_config['training']['learning_rate'])

# Create an instance of the Trainer
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  config_path=config_path,
                  checkpoint_dir="checkpoints/",
                  device=device)

# Start the training process
trainer.train()
