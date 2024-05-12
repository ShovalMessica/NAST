import yaml
from typing import List, Dict, Any


def load_config(*config_paths: str) -> Dict[str, Any]:
    """
    Load and merge multiple configuration files.

    Args:
        *config_paths (str): One or more paths to the configuration files.

    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    merged_config = {}

    for config_path in config_paths:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            merged_config = merge_configs(merged_config, config)

    return merged_config


def merge_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        config1 (Dict[str, Any]): First configuration dictionary.
        config2 (Dict[str, Any]): Second configuration dictionary.

    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    merged_config = config1.copy()

    for key, value in config2.items():
        if isinstance(value, dict):
            if key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key] = merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
        else:
            merged_config[key] = value

    return merged_config
