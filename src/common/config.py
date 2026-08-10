"""Central path and split configuration for FraudShield.

Every module resolves its paths through this file so the same code runs
unchanged in three places:

    local Windows   FRAUDSHIELD_DATA unset      -> <repo>/data
    Colab + Drive   FRAUDSHIELD_DATA=/content/drive/MyDrive/fraudshield/data
    CI / container  FRAUDSHIELD_DATA=/mnt/data

The raw IEEE-CIS CSVs are ~700 MB and never live in git (see .gitignore),
so on Colab the data root points at mounted Drive while the code itself is
cloned fresh. Nothing outside this module reads FRAUDSHIELD_DATA directly.

Usage:
    from src.common.config import TRAIN_PARQUET, SPLITS_DIR, ensure_dirs

    ensure_dirs()
    df = pd.read_parquet(TRAIN_PARQUET)

Run `python -m src.common.config` to print the resolved paths — the fastest
way to confirm a Colab mount landed where you think it did.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------

#: Repository root — this file is <repo>/src/common/config.py, so up three.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Data root. Override with FRAUDSHIELD_DATA to point at mounted Drive,
#: an external disk, or a container volume. Relative values resolve against
#: the repo root, not the current working directory, so notebooks in
#: notebooks/ and scripts in src/ agree on where the data is.
_data_env = os.environ.get("FRAUDSHIELD_DATA", "").strip()
if _data_env:
    _data_path = Path(_data_env).expanduser()
    DATA_ROOT: Path = _data_path if _data_path.is_absolute() else (REPO_ROOT / _data_path)
else:
    DATA_ROOT = REPO_ROOT / "data"
DATA_ROOT = DATA_ROOT.resolve()

# --------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------

#: Untouched downloads: train_transaction.csv, train_identity.csv, Sparkov CSVs.
RAW_DIR: Path = DATA_ROOT / "raw"

#: Temporally split, feature-ready parquet — the inputs to every later stage.
SPLITS_DIR: Path = DATA_ROOT / "splits"

#: Serialised models and their frozen PSI bin edges. Not under DATA_ROOT:
#: artifacts belong with the code that produced them, not the input data.
MODELS_DIR: Path = REPO_ROOT / "models"

#: Drift reports, evaluation output, generated figures.
REPORTS_DIR: Path = REPO_ROOT / "reports"

# --------------------------------------------------------------------------
# Temporal split
# --------------------------------------------------------------------------

#: IEEE-CIS TransactionDT is a seconds offset from an unknown epoch.
SECONDS_PER_DAY: int = 86_400

#: Day boundary between the training window and the replayed "live" window.
#: Days 0-90 train the model and freeze the PSI reference distribution;
#: days 91+ are streamed through Kafka as if arriving in real time, so any
#: distribution shift across this line is the drift the monitor must catch.
#: Revisit once the drift notebook reports actual per-feature PSI.
SPLIT_DAY: int = 91

# --------------------------------------------------------------------------
# The three splits
# --------------------------------------------------------------------------

#: Days 0-90. Fits the XGBoost/LightGBM ensemble.
TRAIN_PARQUET: Path = SPLITS_DIR / "train.parquet"

#: The frozen PSI baseline — the "expected" distribution in the PSI formula.
#: Bin edges are computed here once and versioned with the model; recomputing
#: them on live data makes PSI read ~0 forever and the monitor never fires.
REFERENCE_PARQUET: Path = SPLITS_DIR / "reference.parquet"

#: Days 91+. Replayed as the live transaction stream, and — because the
#: isFraud label is present — doubles as the post-drift evaluation set.
STREAM_PARQUET: Path = SPLITS_DIR / "stream.parquet"

#: Iteration order for code that processes all three.
SPLIT_PATHS: dict[str, Path] = {
    "train": TRAIN_PARQUET,
    "reference": REFERENCE_PARQUET,
    "stream": STREAM_PARQUET,
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create every project directory that does not yet exist.

    Safe to call repeatedly; call it at the top of any script that writes.
    """
    for directory in (RAW_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Return the resolved configuration, with an existence mark per path."""
    source = f"FRAUDSHIELD_DATA={_data_env}" if _data_env else "FRAUDSHIELD_DATA unset (default)"
    rows = [
        ("REPO_ROOT", REPO_ROOT),
        ("DATA_ROOT", DATA_ROOT),
        ("RAW_DIR", RAW_DIR),
        ("SPLITS_DIR", SPLITS_DIR),
        ("MODELS_DIR", MODELS_DIR),
        ("REPORTS_DIR", REPORTS_DIR),
        ("TRAIN_PARQUET", TRAIN_PARQUET),
        ("REFERENCE_PARQUET", REFERENCE_PARQUET),
        ("STREAM_PARQUET", STREAM_PARQUET),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [f"FraudShield config  [{source}]", f"split day: {SPLIT_DAY}", ""]
    lines += [f"  {name:<{width}}  {'x' if path.exists() else ' '}  {path}" for name, path in rows]
    lines.append("")
    lines.append("  (x = exists)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
