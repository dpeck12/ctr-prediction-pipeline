"""Deep & Cross Network v2 (Wang et al., 2021), a standard deep CTR architecture.

Each field's value gets a learned embedding. A cross network then builds explicit
feature interactions (x_{l+1} = x0 * (W x_l + b) + x_l), and an MLP on top learns
whatever the cross layers miss.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn
from torch.nn import functional as F

from ctr.encoders import VocabEncoder, make_fields, record_fields
from ctr.schema import LABEL

log = logging.getLogger(__name__)


class DCNv2(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        embed_dim: int = 16,
        n_cross: int = 3,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        offsets = np.concatenate([[0], np.cumsum(cardinalities)[:-1]])
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(int(sum(cardinalities)), embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

        width = len(cardinalities) * embed_dim
        self.cross = nn.ModuleList(nn.Linear(width, width) for _ in range(n_cross))
        layers: list[nn.Module] = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Integer codes of shape (batch, fields) -> click logits of shape (batch,)."""
        x0 = self.embedding(codes + self.offsets).flatten(1)
        x = x0
        for layer in self.cross:
            x = x0 * layer(x) + x
        return self.head(self.deep(x)).squeeze(-1)


class DCNModel:
    """Trains DCNv2 with Adam, evaluating validation log loss after every epoch.

    CTR models tend to overfit after one or two passes over the data, so training
    stops once validation log loss fails to improve for `patience` epochs and the
    best epoch's weights are restored.
    """

    name = "dcn"

    def __init__(
        self,
        min_count: int = 5,
        embed_dim: int = 16,
        n_cross: int = 3,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        batch_size: int = 4096,
        lr: float = 1e-3,
        max_epochs: int = 5,
        patience: int = 1,
        seed: int = 0,
    ) -> None:
        self.min_count = min_count
        self.arch = {
            "embed_dim": embed_dim,
            "n_cross": n_cross,
            "hidden": tuple(hidden),
            "dropout": dropout,
        }
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.history: list[dict[str, float]] = []

    def _encode(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.from_numpy(self.encoder.transform(make_fields(df)))

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> DCNModel:
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)

        self.encoder = VocabEncoder(self.min_count).fit(make_fields(train))
        x_train, x_valid = self._encode(train), self._encode(valid)
        y_train = torch.tensor(train[LABEL].to_numpy(), dtype=torch.float32)
        y_valid = valid[LABEL].to_numpy()

        self.net = DCNv2(self.encoder.cardinalities, **self.arch)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        best_loss, best_state, stale = float("inf"), None, 0
        self.history = []
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            self.net.train()
            order = torch.randperm(len(x_train), generator=generator)
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                loss = F.binary_cross_entropy_with_logits(self.net(x_train[idx]), y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)

            valid_loss = float(log_loss(y_valid, self._predict(x_valid), labels=[0, 1]))
            self.history.append(
                {"epoch": epoch, "train_log_loss": total / len(order), "valid_log_loss": valid_loss}
            )
            log.info(
                "DCN epoch %d: train %.5f, valid %.5f (%.0fs)",
                epoch,
                total / len(order),
                valid_loss,
                time.perf_counter() - t0,
            )
            if valid_loss < best_loss:
                best_loss, best_state, stale = valid_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        assert best_state is not None
        self.net.load_state_dict(best_state)
        self.best_epoch = min(self.history, key=lambda h: h["valid_log_loss"])["epoch"]
        return self

    @torch.no_grad()
    def _predict(self, codes: torch.Tensor) -> np.ndarray:
        self.net.eval()
        batches = [
            torch.sigmoid(self.net(codes[start : start + 65536]))
            for start in range(0, len(codes), 65536)
        ]
        return torch.cat(batches).numpy().astype(np.float64)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict(self._encode(df))

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        codes = self.encoder.transform_records([record_fields(r) for r in records])
        return self._predict(torch.from_numpy(codes))

    def params(self) -> dict[str, Any]:
        params = {
            "min_count": self.min_count,
            **self.arch,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
        }
        if self.history:
            params["best_epoch"] = self.best_epoch
            params["embedding_rows"] = int(sum(self.encoder.cardinalities))
        return params
