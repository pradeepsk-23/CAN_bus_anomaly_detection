"""
src/models/lstm_autoencoder.py

LSTM Autoencoder for temporal anomaly detection on multi-channel CAN bus signals.

Architecture rationale:
  - Bidirectional option disabled by default: real-time inference on edge requires
    causal models; no future frames available at inference time.
  - Encoder compresses (T, C) → latent vector; decoder reconstructs (T, C).
  - Anomaly score = mean squared reconstruction error over the window.
  - Latent size is the key bottleneck — set tight enough to force the model to
    learn normal dynamics, yet loose enough to handle normal variability.
  - Residuals (per channel, per timestep) are preserved for the signal correlator.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        latent_size: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # Project final hidden state → latent vector
        self.fc = nn.Linear(hidden_size * self.num_directions, latent_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        x: (B, T, C)
        Returns latent: (B, latent_size), hidden state for decoder init
        """
        out, (h_n, c_n) = self.lstm(x)
        # Take final hidden state of last layer (all directions concatenated)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        latent = self.activation(self.fc(h_last))  # (B, latent_size)
        return latent, (h_n, c_n)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        seq_len: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Expand latent → LSTM hidden size
        self.fc_expand = nn.Linear(latent_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, latent_size)
        Returns reconstruction: (B, T, C)
        """
        B = latent.size(0)

        # Repeat latent across time → decoder input
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, latent_size)

        # Initialise hidden state from latent
        h0 = self.fc_expand(latent).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (layers, B, H)
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(decoder_input, (h0, c0))                    # (B, T, H)
        reconstruction = self.output_projection(out)                    # (B, T, C)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder.

    Training: minimise MSE(x, x_hat) on normal data only.
    Inference: anomaly_score(x) = mean(MSE(x, x_hat)) per window.
               High score → reconstruction failure → anomaly.
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        latent_size: int = 16,
        dropout: float = 0.2,
        bidirectional: bool = False,
        seq_len: int = 50,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_size=latent_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.decoder = LSTMDecoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,
            seq_len=seq_len,
            dropout=dropout,
        )

        self._log_parameter_count()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C)
        Returns:
            reconstruction (B, T, C)
            latent         (B, latent_size)
        """
        latent, _ = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    # ------------------------------------------------------------------
    # Anomaly scoring API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-window mean squared reconstruction error.
        x: (B, T, C) → returns (B,) anomaly scores
        """
        self.eval()
        reconstruction, _ = self.forward(x)
        # MSE averaged over T and C dimensions → scalar per window
        error = ((x - reconstruction) ** 2).mean(dim=(1, 2))  # (B,)
        return error

    @torch.no_grad()
    def channel_residuals(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-channel, per-timestep absolute residuals — used by signal correlator
        to identify WHICH channels drove an anomaly detection.
        x: (B, T, C) → returns (B, T, C)
        """
        self.eval()
        reconstruction, _ = self.forward(x)
        return (x - reconstruction).abs()

    def _log_parameter_count(self) -> None:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("LSTMAutoencoder — {:,} trainable parameters", n)

    @classmethod
    def from_config(cls, cfg: dict, seq_len: int) -> "LSTMAutoencoder":
        m = cfg["models"]["lstm_autoencoder"]
        return cls(
            input_size=m["input_size"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            latent_size=m["latent_size"],
            dropout=m["dropout"],
            bidirectional=m["bidirectional"],
            seq_len=seq_len,
        )
