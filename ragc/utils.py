from pathlib import Path
import random
import os

import yaml
import numpy as np
import torch
from pydantic import BaseModel


def load_config(cls: type[BaseModel], cfg_p: Path | str) -> BaseModel:
    cfg_p = Path(cfg_p)
    with cfg_p.open("r") as f:
        data = yaml.safe_load(f)

    return cls.model_validate(data)


def save_config(model: BaseModel, cfg_p: Path | str) -> None:
    """Save a Pydantic BaseModel instance to a YAML file.

    Args:
        model: The Pydantic model instance to save.
        cfg_p: The path to the YAML file where the model will be saved.

    """
    cfg_p = Path(cfg_p)
    # Convert the model to a dictionary
    model_dict = model.model_dump(mode="json")

    # Write the dictionary to a YAML file
    with cfg_p.open("w") as f:
        yaml.safe_dump(model_dict, f, default_flow_style=False, sort_keys=False)


def fix_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility across."""
    # Python random
    random.seed(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
