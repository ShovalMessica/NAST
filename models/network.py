import torch
from torch import nn
import torchaudio
import torch.nn.functional as F
from typing import Dict, Any


class Network(nn.Module):
    def __init__(self, config: Dict[str, Any], device: torch.device = torch.device("cpu")):
        super().__init__()
        self.config = config
        self.device = device
        self.num_units = config["num_units"]
        self.in_channels = config[self.num_units]["in_channels"]
        self.vocab_size = config["num_units"]
        self.discrete_local = config[self.num_units]["discrete_local"]
        self.out_channels = 768 if config[self.num_units]["reconstruction_type"] == "HuBERT" else None
        self.continuous_global = config[self.num_units]["continuous_global"]

        self.predictor = self._build_conformer(
            input_dim=self.in_channels,
            num_heads=config[self.num_units]["predictor"]["num_heads"],
            ffn_dim=config[self.num_units]["predictor"]["ffn_dim"],
            num_layers=config[self.num_units]["predictor"]["num_layers"],
            depthwise_conv_kernel_size=config[self.num_units]["predictor"]["depthwise_conv_kernel_size"]
        ).to(self.device)

        self.projector = nn.Linear(self.in_channels, self.num_units).to(self.device)

        self.residual_encoder = self._build_conformer(
            input_dim=self.in_channels,
            num_heads=config[self.num_units]["residual_encoder"]["num_heads"],
            ffn_dim=config[self.num_units]["residual_encoder"]["ffn_dim"],
            num_layers=config[self.num_units]["residual_encoder"]["num_layers"],
            depthwise_conv_kernel_size=config[self.num_units]["residual_encoder"]["depthwise_conv_kernel_size"]
        ).to(self.device)

        self.residual_projector = nn.Linear(768, 256).to(self.device)

        self.decoder = self._build_conformer(
            input_dim=self.num_units + 256,
            num_heads=config[self.num_units]["decoder"]["num_heads"],
            ffn_dim=config[self.num_units]["decoder"]["ffn_dim"],
            num_layers=config[self.num_units]["decoder"]["num_layers"],
            depthwise_conv_kernel_size=config[self.num_units]["decoder"]["depthwise_conv_kernel_size"]
        ).to(self.device)

        self.decoder_projection = nn.Linear(self.num_units + 256, self.out_channels).to(self.device)

    def _build_conformer(self, input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size):
        return torchaudio.models.Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size
        )

    def _get_residual_information(self, input_features):
        residual_information = self.residual_encoder(torch.unsqueeze(input_features, dim=0),
                                                     torch.tensor([input_features.shape[0]]).to(input_features.device))
        residual_information = self.residual_projector(torch.squeeze(residual_information[0], dim=0))
        residual_information = torch.mean(residual_information, 0, keepdim=True)
        residual_information = residual_information.repeat(input_features.shape[0], 1).requires_grad_()
        return residual_information

    def _get_predicts(self, input_features):
        predicts = self.predictor(torch.unsqueeze(input_features, dim=0),
                                  torch.tensor([input_features.shape[0]]).to(input_features.device))
        predicts = self.projector(torch.squeeze(predicts[0], dim=0))
        return predicts

    def forward(self, input_features: torch.Tensor):
        """
        Args:
            input_features (torch.Tensor): Input features tensor.

        Returns:
            - torch.Tensor: The reconstructed representation of the input.
            - torch.Tensor: The unit selection as a one-hot vector.
            - torch.Tensor: The logits before sampling.
        """
        predicts = self._get_predicts(input_features)
        one_hot_vector = F.gumbel_softmax(logits=predicts, tau=0.8, hard=False, dim=1)
        residual_information = self._get_residual_information(input_features)

        if self.discrete_local:
            return torch.argmax(one_hot_vector, dim=1)

        if self.continuous_global:
            return residual_information

        x = torch.cat([one_hot_vector, residual_information], 1)

        output = self.decoder(torch.unsqueeze(x, dim=0), torch.tensor([x.shape[0]]).to(x.device))
        output = self.decoder_projection(torch.squeeze(output[0], dim=0))

        return output, one_hot_vector, predicts
