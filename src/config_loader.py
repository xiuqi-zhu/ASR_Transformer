"""Config loader module."""

import os
import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load YAML config file.

    Args:
        config_path: Path to config file, defaults to config/config.yaml.

    Returns:
        Config dictionary.
    """
    if config_path is None:
        # Default path: config/config.yaml under project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config
