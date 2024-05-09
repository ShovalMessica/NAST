from typing import Tuple, Dict, Any
import logging

import torch
import torch.nn.functional as F

from fairseq.data.audio.audio_utils import get_features_or_waveform


def adjust_cross_entropy_weight(ce_loss_weight: float, ce_loss: torch.Tensor, ce_loss_prev: torch.Tensor,
                                ce_loss_stabilized: bool, config: Dict[str, Any]) -> Tuple[float, torch.Tensor, bool]:
    """
    Adjust the cross-entropy loss weight based on the stabilization threshold and increment factor.

    Args:
        ce_loss_weight (float): Cross-entropy loss weight.
        ce_loss (torch.Tensor): Current cross-entropy loss value.
        ce_loss_prev (torch.Tensor): Previous cross-entropy loss value.
        ce_loss_stabilized (bool): Flag indicating if the cross-entropy loss has stabilized.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[float, torch.Tensor, bool]: Updated cross-entropy loss weight, previous cross-entropy loss value, and stabilization flag.
    """
    if not ce_loss_stabilized:
        if abs(ce_loss - ce_loss_prev) < config['cross_entropy']['stabilization_threshold']:
            ce_loss_stabilized = True
        else:
            ce_loss_weight = min(ce_loss_weight * config['cross_entropy']['weight_increment_factor'],
                                 config['cross_entropy']['max_weight'])
            ce_loss_prev = ce_loss
    return ce_loss_weight, ce_loss_prev, ce_loss_stabilized


def synchronize_diversity_weight(diversity_loss: torch.Tensor, diversity_prev: torch.Tensor, diversity_weight: float,
                                 config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synchronize the diversity weight based on the current and previous diversity loss values.

    Args:
        diversity_loss (torch.Tensor): Current diversity loss value.
        diversity_prev (torch.Tensor): Previous diversity loss value.
        diversity_weight (float): Current diversity weight.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated diversity weight and previous diversity loss value.
    """
    if diversity_loss < diversity_prev:
        diversity_weight = min(diversity_weight * config['diversity']['synchronization_factor'],
                               config['diversity']['max_weight'])
    diversity_prev = diversity_loss
    return diversity_weight, diversity_prev


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

def read_audio(feature_extractor, path, ref_len=None):
    wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=feature_extractor.task.cfg.sample_rate)
    if wav.ndim == 2:
        wav = wav.mean(-1)
    assert wav.ndim == 1, wav.ndim
    if ref_len is not None and abs(ref_len - len(wav)) > 160:
        logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
    return wav


def get_feats(feature_extractor, x):
    with torch.no_grad():
        x = torch.from_numpy(x).float().cuda()
        if feature_extractor.task.cfg.normalize:
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)

        feat = []
        for start in range(0, x.size(1), feature_extractor.max_chunk):
            x_chunk = x[:, start: start + feature_extractor.max_chunk]
            feat_chunk, _ = feature_extractor.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=feature_extractor.layer,
            )
            feat.append(feat_chunk)
    return torch.cat(feat, 1).squeeze(0)
