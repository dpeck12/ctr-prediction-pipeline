"""A minimal model registry: versioned bundles on disk and a pointer to the champion.

The champion is the version being served. Each day's newly trained challenger
only replaces it if it does at least as well on held-out data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ctr.serving.bundle import ModelBundle

INDEX = "registry.json"


class Registry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _read(self) -> dict[str, Any]:
        path = self.root / INDEX
        if path.exists():
            return json.loads(path.read_text())
        return {"champion": None, "versions": {}}

    def _write(self, index: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / INDEX).write_text(json.dumps(index, indent=2))

    def register(self, version: str, bundle: ModelBundle, **info: Any) -> Path:
        """Save a bundle as `version`. Registering the same version again replaces it."""
        path = bundle.save(self.root / version)
        index = self._read()
        index["versions"][version] = {
            "path": path.name,  # relative to the registry root, so it can be moved
            "model": bundle.name,
            "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **info,
        }
        self._write(index)
        return path

    def versions(self) -> dict[str, dict[str, Any]]:
        return self._read()["versions"]

    def info(self, version: str) -> dict[str, Any]:
        return self.versions()[version]

    @property
    def champion(self) -> str | None:
        return self._read()["champion"]

    def promote(self, version: str) -> None:
        index = self._read()
        if version not in index["versions"]:
            raise KeyError(f"Unknown version {version}")
        index["champion"] = version
        self._write(index)

    def load(self, version: str) -> ModelBundle:
        return ModelBundle.load(self.root / version)
