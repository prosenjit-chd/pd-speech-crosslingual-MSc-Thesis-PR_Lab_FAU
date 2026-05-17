import os
import yaml
import logging
import random
import numpy as np
import torch

def setup_logging(level=logging.INFO):
    """Sets up a basic logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def load_config(config_path="configs/baseline_config.yaml"):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

from pathlib import Path

def ensure_dir(path):
    """Creates directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_project_directories():
    """Initializes the required project directories."""
    directories = [
        "metadata",
        "features",
        "features/xlsr",
        "features/wav2vec2",
        "features/wavlm",
        "outputs",
        "outputs/figures",
        "outputs/tables",
        "outputs/reports",
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info("Ensured all project directories exist.")

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Returns the available torch device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.getLogger(__name__).info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).warning("GPU not available. Using CPU. Extraction may be slow.")
    return device
