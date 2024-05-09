import os
import torch


def save_checkpoint(model: torch.nn.Module, epoch: int, batch_idx: int, checkpoint_dir: str, is_best: bool) -> None:
    """
    Save the model's state dictionary to a checkpoint file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        epoch (int): The current epoch number.
        batch_idx (int): The current batch index within the epoch.
        checkpoint_dir (str): The directory where the checkpoint files will be saved.
        is_best (bool): True if the model has the best performance so far, False otherwise.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_name = 'best_model.pt' if is_best else f'epoch_{epoch}_batch_{batch_idx}.pt'
    save_dict = model.state_dict()
    checkpoint_path = os.path.join(checkpoint_dir, save_name)
    try:
        torch.save(save_dict, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    except IOError as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load the model's weights from a given checkpoint.

    Args:
        model (torch.nn.Module): Model to load the checkpoint into.
        ckpt_path (str): Path to the checkpoint file.
    """
    state_dict = torch.load(ckpt_path)

    # Check if the model was saved with nn.DataParallel
    if list(state_dict.keys())[0].startswith("module."):
        # Create a new state_dict without the module. prefix
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
