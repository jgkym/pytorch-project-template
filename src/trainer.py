import copy
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        self.model = model

        # Handle multi-GPU training with DataParallel
        self.is_distributed = torch.cuda.device_count() > 1 and "cuda" in str(
            self.device
        )
        if self.is_distributed:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.model = nn.DataParallel(self.model)

        self.model.to(device=self.device)

        self.use_amp = "cuda" in str(self.device)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.min_lr,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self.best_model_state = None
        self.best_loss = float("inf")

        self.early_stopping_patience = config.early_stopping_patience
        self.patience_counter = 0

        # Ensure output directory exists
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config.do_log:
            wandb.init(
                project=config.project_name,
                config={
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                },
            )

    def train(self):
        """Main training loop."""

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(epoch)

            current_loss = train_loss
            log_message = f"Epoch {epoch + 1}/{self.config.num_epochs}\nTrain Loss: {train_loss:.4f}"

            if self.val_loader:
                eval_metrics = self._evaluate()
                current_loss = eval_metrics["loss"]
                log_message += f"\nVal Loss: {eval_metrics['loss']:.4f} | Val Acc: {eval_metrics['acc']:.4f} | Val F1: {eval_metrics['f1']:.4f}"

            # The scheduler should be stepped once per epoch
            self.scheduler.step()

            if current_loss < self.best_loss:
                self.patience_counter = 0
                self.best_loss = current_loss
                log_message += f"\nNew best loss: {self.best_loss:.4f}. Saving model..."

                # Save the best model state to disk
                model_state = (
                    self.model.module.state_dict()
                    if self.is_distributed
                    else self.model.state_dict()
                )
                self.best_model_state = copy.deepcopy(model_state)
                torch.save(model_state, self.output_dir / "best_model.pth")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    log_message += "\nEarly stopping triggered."
                    print(log_message)
                    break
            if self.config.do_log:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": eval_metrics["loss"],
                        "best_loss": self.best_loss,
                    }
                )
            print(log_message)

        print(f"\n\nTraining completed. Best loss: {self.best_loss:.4f}")
        if self.config.do_log:
            wandb.finish()

        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        elif "mps" in str(self.device):
            torch.mps.empty_cache()

        return self.best_model_state

    def _train_epoch(self, epoch):
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            leave=False,
        )

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = (
                inputs.to(self.device, non_blocking=True),
                labels.to(self.device, non_blocking=True),
            )
            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(str(self.device), enabled=self.use_amp):
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                if self.config.do_log and i % self.config.logging_steps == 0:
                    wandb.log({"running_loss": loss.item()})

            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # Update scaler
            self.scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.1e}"
            )

        return running_loss / len(self.train_loader)

    def _evaluate(self):
        """Runs a single evaluation epoch."""
        self.model.eval()
        val_loss = 0.0
        progress_bar = tqdm(
            self.val_loader,
            total=len(self.val_loader),
            desc="Evaluating",
            leave=False,
        )

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = (
                    inputs.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True),
                )
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        loss = val_loss / len(self.val_loader)

        return {
            "loss": loss,
        }
