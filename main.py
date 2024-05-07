import yaml
import torch
from fairseq.examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
from models.network import Network
from utils.checkpoint import load_checkpoint

model_config_path = "/path/to/model_config.yaml"
training_config_path = "/path/to/training_config.yaml"
audio_path = "/path/to/audio.wav"

with open(model_config_path, "r") as f:
    model_config = yaml.safe_load(f)

with open(training_config_path, "r") as f:
    training_config = yaml.safe_load(f)

config = {**model_config, **training_config}

config["discrete_local"] = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = HubertFeatureReader(config['checkpoints']['hubert'], layer=9)
network = Network(config=config, device=device)

# Load checkpoint if available
if config['checkpoints']['ulm']:
    load_checkpoint(network, config['checkpoints']['ulm'])

audio = feature_extractor.read_audio(audio_path)
features = feature_extractor.get_feats(audio)

with torch.no_grad():
    units = network(features.to(device))

print("Extracted units:", units.tolist())