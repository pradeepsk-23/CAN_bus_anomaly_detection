"""
src/training/trainer.py

Production-grade trainer for the LSTM Autoencoder.

Features:
  - Trains on normal windows only (unsupervised — no labels used)
  - Early stopping on validation reconstruction loss
  - LR scheduling (cosine / step / plateau)
  - Gradient clipping
  - TensorBoard logging
  - Checkpoint saving (best + latest)
  - Mixed-precision support (torch.cuda.amp)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.data.dataset import WindowDataset


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered (patience={}, best_val_loss={:.6f})",
                            self.patience, self.best_loss)
        return self.should_stop


class LSTMAETrainer:
    """
    Trains an LSTM Autoencoder in a fully unsupervised fashion.
    Only normal (non-anomalous) windows are used during training.
    """

    def __init__(
        self,
        model: LSTMAutoencoder,
        cfg: dict,
        device: torch.device,
        checkpoint_dir: str = "artifacts/checkpoints",
        tensorboard_dir: str = "artifacts/tensorboard",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["training"]
        self.max_epochs = train_cfg["max_epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.batch_size = train_cfg["batch_size"]
        self.grad_clip = train_cfg["gradient_clip_norm"]
        self.num_workers = train_cfg["num_workers"]
        self.mixed_precision = train_cfg["mixed_precision"] and device.type == "cuda"

        self.optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = self._build_scheduler(train_cfg)
        self.early_stopping = EarlyStopping(
            patience=train_cfg["early_stopping"]["patience"],
            min_delta=train_cfg["early_stopping"]["min_delta"],
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Public training entry point
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: "np.ndarray",
        y_train: "np.ndarray",
        X_val: "np.ndarray",
        y_val: "np.ndarray",
    ) -> Dict[str, list]:
        """
        X_train / X_val: (N, T, C) normalised windows
        y_train / y_val: (N,) labels — used ONLY to filter out anomalies from training

        Returns history dict with train_loss and val_loss per epoch.
        """
        import numpy as np

        # -- Filter: train on NORMAL windows only
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask]
        logger.info("Training on {:,} normal windows ({:.1%} of train set)",
                    X_train_normal.shape[0], normal_mask.mean())

        train_loader = self._build_loader(X_train_normal, shuffle=True)
        val_loader = self._build_loader(X_val, shuffle=False)   # val includes anomalies

        history = {"train_loss": [], "val_loss": []}
        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._val_epoch(val_loader, epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # TensorBoard
            self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # LR scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt", epoch, val_loss)

            # Early stopping
            if self.early_stopping.step(val_loss):
                logger.info("Training stopped at epoch {}/{}", epoch, self.max_epochs)
                break

            if epoch % 5 == 0 or epoch == 1:
                elapsed = time.time() - t0
                logger.info(
                    "Epoch {:4d}/{} | train={:.6f} | val={:.6f} | lr={:.2e} | {:.1f}s elapsed",
                    epoch, self.max_epochs, train_loss, val_loss,
                    self.optimizer.param_groups[0]["lr"], elapsed,
                )

        self._save_checkpoint("final_model.pt", epoch, val_loss)
        self.writer.close()
        logger.info("Training complete. Best val_loss={:.6f}", self.best_val_loss)
        return history

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                reconstruction, _ = self.model(x)
                loss = self.criterion(reconstruction, x)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                reconstruction, _ = self.model(x)
                loss = self.criterion(reconstruction, x)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_loader(self, X: "np.ndarray", shuffle: bool) -> DataLoader:
        dataset = WindowDataset(X)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def _build_scheduler(self, train_cfg: dict):
        sched_cfg = train_cfg["lr_scheduler"]
        sched_type = sched_cfg["type"]
        if sched_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg["t_max"],
                eta_min=sched_cfg["min_lr"],
            )
        elif sched_type == "plateau":
            return ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        elif sched_type == "step":
            return StepLR(self.optimizer, step_size=20, gamma=0.5)
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)
        logger.debug("Checkpoint saved → {} (epoch={}, val_loss={:.6f})", path, epoch, val_loss)
