from pathlib import Path

from databento_pipeline import TIMEZONE, create_historical_client, load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Repository directory
MAX_EVENTS = 250000  # Maximum events accepted per in-memory run
SUPPORTED_SCHEMAS = ("trades", "mbp-1")  # Supported market schemas
SUPPORTED_DATASETS = ("XNAS.ITCH", "XNYS.PILLAR", "XASE.PILLAR", "ARCX.PILLAR", "BATS.PITCH", "XBOS.ITCH")  # Supported equity feeds
