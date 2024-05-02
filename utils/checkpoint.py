import torch

def save_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Save the model's state dictionary to a checkpoint file.

    Args:
        model (torch.nn.Module): Model to save.
        ckpt_path (str): Path to save the checkpoint.
    """
    torch.save(model.state_dict(), ckpt_path)

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
