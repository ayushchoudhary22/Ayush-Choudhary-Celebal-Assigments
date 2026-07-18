"""
Configuration Management Utility Module.
Provides type-safe access to project parameters defined in configs/config.yaml.
Preserves legacy constant exports for full backward compatibility.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

# Base directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = RAW_DATA_DIR / "sample_submission.csv"

DATE_COL = "date"
STORE_COL = "store"
ITEM_COL = "item"
TARGET_COL = "sales"

LAG_DAYS = [1, 7, 14, 30, 60, 90]
ROLLING_WINDOWS = [7, 14, 30]

DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_SERVICE_LEVEL = 0.95

RANDOM_SEED = 42


class ConfigLoader:
    """Singleton Configuration Loader for RetailSense AI Platform."""

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._config is None:
            self.load_config()

    def get_config_path(self) -> Path:
        """Resolves absolute path to config.yaml."""
        config_path = BASE_DIR / "configs" / "config.yaml"
        if not config_path.exists():
            # Fallback if config.yaml does not exist
            return config_path
        return config_path

    def load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file safely."""
        config_path = self.get_config_path()
        if not config_path.exists():
            self._config = {}
            return self._config
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            return self._config
        except Exception as e:
            self._config = {}
            return self._config

    @property
    def config(self) -> Dict[str, Any]:
        """Returns active configuration dictionary."""
        if self._config is None:
            return self.load_config()
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """Safely retrieves nested configuration value using dot notation."""
        keys = key_path.split(".")
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val


def get_config() -> Dict[str, Any]:
    """Helper function returning global configuration dictionary."""
    return ConfigLoader().config


def get_setting(key_path: str, default: Any = None) -> Any:
    """Helper function for retrieving nested setting using dot notation."""
    return ConfigLoader().get(key_path, default)
