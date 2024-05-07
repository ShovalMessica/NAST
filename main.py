import yaml
import torch
from fairseq.examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
from models.network import Network

config_path = "/path/to/config.yaml"
audio_path = "/path/to/audio.wav"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["discrete_local"] = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = HubertFeatureReader(config['checkpoints']['hubert'], layer=9)
network = Network(config=config, device=device)

audio = feature_extractor.read_audio(audio_path)
features = feature_extractor.get_feats(audio)

with torch.no_grad():
    units = network(features.to(device))

print("Extracted units:", units.tolist()) # [10 11 11 11 21 32 32 32 21]