import yaml

def load_config(model_name: str) -> Dict[str, Any]:
    """
    Load the configuration for the specified model.

    Args:
        model_name (str): Name of the model configuration.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.
    """
    with open(f"config/{model_name}.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
