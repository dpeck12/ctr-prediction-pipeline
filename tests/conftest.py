import pytest

from ctr.sample import make_synthetic


@pytest.fixture
def raw_csv(tmp_path):
    """A small synthetic file in the raw Avazu format, gzipped like the real one."""
    path = tmp_path / "train.gz"
    make_synthetic(n_rows=2_000, n_days=10, seed=1).to_csv(path, index=False, compression="gzip")
    return path
