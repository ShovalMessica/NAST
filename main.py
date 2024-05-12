import torch
from fairseq.examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
from utils.training_utils import read_audio, get_feats
from models.network import Network
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def main():
    model_config_path = "path/to/model_config.yaml"
    audio_path = "path/to/audio/file.wav"
    num_units = 100

    model_config = load_config(model_config_path)
    config = {**model_config, 'num_units': num_units}

    config['hubert']['checkpoint_path'] = "path/to/hubert/checkpoint.pt"

    config[num_units]["discrete_local"] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = HubertFeatureReader(config['hubert']['checkpoint_path'], layer=9)
    network = Network(config=config, device=device)

    load_checkpoint(network, "path/to/tokenizer/checkpoint")

    audio = read_audio(feature_extractor, audio_path)
    features = get_feats(feature_extractor, audio)

    with torch.no_grad():
        units = network(features.to(device))

    print("Extracted units:", units.tolist())


if __name__ == "__main__":
    main()
