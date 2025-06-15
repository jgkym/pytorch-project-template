from src.models.base import BaseModel
from src.utils import set_seed
from src.configs.base import Config
from torch.utils.data import Dataset, DataLoader

from trainer import Trainer


def prepare_dataloaders(config) -> tuple[DataLoader, DataLoader | None]:
    """
    Handles all data preparation: loading, undersampling, splitting, and creating DataLoaders.

    Returns:
        A tuple of (train_loader, val_loader). val_loader can be None.
    """
    print("Preparing dataloaders...")
    # Load base dataset
    base_dataset: Dataset

    # Prepare train and validation datasets
    if config.split_ratio > 0:
        train_dataset: Dataset
        val_dataset: Dataset

        print(
            f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        # Use the entire balanced subset for training
        train_dataset = base_dataset
        val_loader = None
        print(f"Train samples: {len(train_dataset)}, No validation split.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train(config):
    """
    Initializes and runs the model training process.
    """
    # --- Seed Setting ---
    set_seed(config)

    # --- Data Preparation ---
    train_loader, val_loader = prepare_dataloaders(config)

    # --- Model and Training ---
    model = BaseModel(config)
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print("Starting training...")
    _ = trainer.train()
    print("Training finished.")


if __name__ == "__main__":
    config = Config()
    train(config)
