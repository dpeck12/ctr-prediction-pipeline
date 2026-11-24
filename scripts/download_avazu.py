"""Download the Avazu train file from Kaggle into data/raw/.

Before running:
  1. Accept the competition rules at https://www.kaggle.com/competitions/avazu-ctr-prediction/rules
  2. Put a Kaggle API token in ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token)
  3. Install the extra: uv sync --extra data

Only train.gz is used. Avazu's test file has no labels, so evaluation uses the
last days of train instead (see ctr.data.split_by_day).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COMPETITION = "avazu-ctr-prediction"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "competitions", "download",
        "-c", COMPETITION, "-f", "train.gz", "-p", str(RAW_DIR),
    ]  # fmt: skip
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
