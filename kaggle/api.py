"""Kaggle API client and authentication utilities."""
import json
import config_local.local_config as config


def get_username() -> str:
    """Get Kaggle username from kaggle.json."""
    kaggle_json_path = config.KAGGLE_JSON
    if kaggle_json_path.exists():
        with open(kaggle_json_path) as f:
            kaggle_config = json.load(f)
            return kaggle_config.get("username", "unknown")
    return "unknown"

