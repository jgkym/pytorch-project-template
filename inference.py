import pandas as pd
import torch
from src.models.base import BaseModel
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.configs.base_config import Config
from src.utils import set_seed


def prepare_testloader(config: Config) -> DataLoader:
    """
    Prepare the test dataloader for inference.
    """
    # --- Dataset and DataLoader ---
    print("Setting up dataset and dataloader...")
    test_dataset: Dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=config.pin_memory,
    )
    return test_loader


def inference(config):
    """
    Run inference on the test dataset, generate predictions,
    and save the results to a submission CSV file.
    """
    # --- Seed Setting ---
    set_seed(config)

    print("Starting inference...")
    print(f"Using device: {config.device}")

    test_loader = prepare_testloader(config)

    # --- Model Loading ---
    model_save_path = config.output_dir / "best_model.pth"
    print(f"Loading model from {model_save_path}...")
    config.pretrained = False
    model = BaseModel(config)
    model.load_state_dict(torch.load(model_save_path, map_location=config.device))
    is_distributed = torch.cuda.device_count() > 1 and "cuda" in str(config.device)
    if is_distributed:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    use_amp = "cuda" in str(config.device)

    model.to(config.device)
    model.eval()

    # --- Inference Loop ---
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Inference", leave=False)
        for inputs in progress_bar:
            inputs = inputs.to(config.device, non_blocking=True)

            # Use mixed precision for faster inference on compatible GPUs
            with autocast(str(config.device), enabled=use_amp):
                outputs = model(inputs)

    # --- Process and Save Results ---
    print("Processing and saving results...")

    pass

    print(f"Inference completed.")


if __name__ == "__main__":
    config = Config()
    inference(config)
