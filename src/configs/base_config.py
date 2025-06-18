import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    # Paths
    base_dir: Path = Path("__file__").resolve().parent
    data_dir: Path = base_dir / "data"
    train_dir: Path = data_dir / "train"
    output_dir: Path = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # Base
    project_name: str = "your-project-name"
    batch_size: int = 64
    split_ratio: float = 0.2
    random_seed: int = 42
    num_workers: int = os.cpu_count()
    pin_memory: bool = True if torch.cuda.is_available() else False

    # Model
    model_name: str = ""

    # Training
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    min_lr: float = 1e-6
    gradient_accumulation_steps: int = 2
    label_smoothing: float = 0.1
    early_stopping_patience: int = 5
    report_to: str | None = "wandb"
    logging_steps: int = 100
