"""Download the Avazu train file from Kaggle into data/raw/.

Before running:
  1. Accept the competition rules at https://www.kaggle.com/competitions/avazu-ctr-prediction/rules
  2. Create a Kaggle API token (Kaggle > Settings > API) and save it to
     ~/.kaggle/access_token (new KGAT_ tokens) or ~/.kaggle/kaggle.json (legacy JSON)
  3. Install the extra: uv sync --extra data

Only train.gz is used. Avazu's test file has no labels, so evaluation uses the
last days of train instead (see ctr.data.split_by_day).
"""

from __future__ import annotations

import sys
from pathlib import Path

COMPETITION = "avazu-ctr-prediction"
FILE = "train.gz"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    # Imported here: the kaggle package authenticates on import and needs the extra.
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {COMPETITION}/{FILE} to {RAW_DIR}")
    api.competition_download_file(COMPETITION, FILE, path=str(RAW_DIR), quiet=False)

    path = RAW_DIR / FILE
    if not path.exists():
        print(f"Download finished but {path} is missing", file=sys.stderr)
        return 1
    print(f"Saved {path} ({path.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
