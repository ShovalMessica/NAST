import torch

def adjust_cross_entropy_weight(ce_loss, ce_loss_prev, ce_loss_stabilized, config):
    if not ce_loss_stabilized:
        if abs(ce_loss - ce_loss_prev) < config['cross_entropy']['stabilization_threshold']:
            ce_loss_stabilized = True
        else:
            ce_loss_weight = min(ce_loss_weight * config['cross_entropy']['weight_increment_factor'], config['cross_entropy']['max_weight'])
            ce_loss_prev = ce_loss
    return ce_loss_weight, ce_loss_prev, ce_loss_stabilized

def synchronize_diversity_weight(diversity_loss, diversity_prev, diversity_weight, config):
    if diversity_loss < diversity_prev:
        diversity_weight = min(diversity_weight * config['diversity']['synchronization_factor'], config['diversity']['max_weight'])
    diversity_prev = diversity_loss
    return diversity_weight, diversity_prev
