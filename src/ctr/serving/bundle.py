"""A trained model saved to disk with the facts needed to trust it in production."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ctr.models import CTRModel

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class ModelBundle:
    """The fitted model (including its encoders) plus metadata.

    Metadata records what the model is, which days it was trained on and how it
    scored on the test day, so the service can report exactly what it's running.
    """

    model: CTRModel
    metadata: dict[str, Any]

    @property
    def name(self) -> str:
        return self.model.name

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / MODEL_FILE)
        (path / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        """Load a bundle. Only load bundles you created: joblib files can run code."""
        path = Path(path)
        model = joblib.load(path / MODEL_FILE)
        metadata = json.loads((path / METADATA_FILE).read_text())
        return cls(model=model, metadata=metadata)
