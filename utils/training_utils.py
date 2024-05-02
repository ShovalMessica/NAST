from typing import Tuple

def adjust_cross_entropy_weight(ce_loss: float, ce_loss_prev: float, ce_loss_stabilized: bool, config: Dict[str, Any]) -> Tuple[float, float, bool]:
    """
    Adjust the cross-entropy loss weight based on the stabilization threshold and increment factor.

    Args:
        ce_loss (float): Current cross-entropy loss value.
        ce_loss_prev (float): Previous cross-entropy loss value.
        ce_loss_stabilized (bool): Flag indicating if the cross-entropy loss has stabilized.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[float, float, bool]: Updated cross-entropy loss weight, previous cross-entropy loss value, and stabilization flag.
    """
    if not ce_loss_stabilized:
        if abs(ce_loss - ce_loss_prev) < config['cross_entropy']['stabilization_threshold']:
            ce_loss_stabilized = True
        else:
            ce_loss_weight = min(ce_loss_weight * config['cross_entropy']['weight_increment_factor'], config['cross_entropy']['max_weight'])
            ce_loss_prev = ce_loss
    return ce_loss_weight, ce_loss_prev, ce_loss_stabilized

def synchronize_diversity_weight(diversity_loss: float, diversity_prev: float, diversity_weight: float, config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Synchronize the diversity weight based on the current and previous diversity loss values.

    Args:
        diversity_loss (float): Current diversity loss value.
        diversity_prev (float): Previous diversity loss value.
        diversity_weight (float): Current diversity weight.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[float, float]: Updated diversity weight and previous diversity loss value.
    """
    if diversity_loss < diversity_prev:
        diversity_weight = min(diversity_weight * config['diversity']['synchronization_factor'], config['diversity']['max_weight'])
    diversity_prev = diversity_loss
    return diversity_weight, diversity_prev
