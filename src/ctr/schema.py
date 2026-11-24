"""Column definitions for the Avazu click-through-rate dataset.

The raw file has one row per ad impression. `click` is the label, `hour` is the
impression time as YYMMDDHH, and every other column is an anonymised categorical
feature (Avazu hashes most of them, so even the numeric-looking C1, C14-C21 are
treated as categories).
"""

ID = "id"
LABEL = "click"
RAW_TIME = "hour"
TIMESTAMP = "timestamp"

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
)

RAW_COLUMNS: tuple[str, ...] = (ID, LABEL, RAW_TIME, *CATEGORICAL_COLUMNS)
