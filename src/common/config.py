"""Central path and split configuration for FraudShield.

Every module resolves its paths through this file so the same code runs
unchanged in three places:

    local Windows   FRAUDSHIELD_DATA unset      -> <repo>/data
    Colab + Drive   FRAUDSHIELD_DATA=/content/drive/MyDrive/fraudshield/data
    CI / container  FRAUDSHIELD_DATA=/mnt/data

The raw IEEE-CIS CSVs and the split parquet are ~700 MB and never live in
git (see .gitignore), so on Colab the data root points at mounted Drive
while the code itself is cloned fresh. Nothing outside this module reads
FRAUDSHIELD_DATA directly.

Split boundaries are NOT defined here. They were fixed when Phase 0 wrote
the parquet files and are recorded in split_manifest.json next to them;
this module reads that file. A constant in the repo could drift out of
step with the data on disk after a re-split, and the resulting bug -- a
PSI reference window that quietly disagrees with the data it baselines --
would be near-invisible. The manifest is the single source of truth.

Usage:
    from src.common.config import TRAIN_PARQUET, split_boundaries

    df = pd.read_parquet(TRAIN_PARQUET)
    bounds = split_boundaries()          # reads + validates the manifest

Run `python -m src.common.config` to print resolved paths and the loaded
boundaries -- the fastest way to confirm a Colab mount landed where you
think it did.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------

#: Repository root -- this file is <repo>/src/common/config.py, so up three.
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

#: Phase 0 output -- the merged, temporally split parquet plus its manifest.
SPLITS_DIR: Path = DATA_ROOT / "splits"

#: Serialised models and their frozen PSI bin edges. Not under DATA_ROOT:
#: artifacts belong with the code that produced them, not the input data.
MODELS_DIR: Path = REPO_ROOT / "models"

#: Drift reports, evaluation output, generated figures.
REPORTS_DIR: Path = REPO_ROOT / "reports"

# --------------------------------------------------------------------------
# The three splits (Phase 0, complete)
# --------------------------------------------------------------------------
#
# IEEE-CIS train_transaction LEFT JOIN train_identity on TransactionID,
# 590,540 rows x 434 columns, cut on TransactionDT into sixths.

#: Sixths 1-3. Fits the ensemble, and is also the frozen PSI reference --
#: the "expected" side of the PSI formula. There is no separate reference
#: split; baselining against the data the model actually learned is the
#: whole point.
TRAIN_PARQUET: Path = SPLITS_DIR / "train.parquet"

#: Sixth 4. Held out for threshold selection, early stopping, and model
#: comparison. Never used to fit, never used as the PSI baseline.
VAL_PARQUET: Path = SPLITS_DIR / "val.parquet"

#: Sixths 5-6. Replayed through Kafka as the live transaction stream, and --
#: because isFraud is present -- doubles as the post-drift evaluation set.
STREAM_PARQUET: Path = SPLITS_DIR / "stream.parquet"

#: Records the exact TransactionDT cut points used to produce the above.
SPLIT_MANIFEST_PATH: Path = SPLITS_DIR / "split_manifest.json"

#: Iteration order for code that processes all three.
SPLIT_PATHS: dict[str, Path] = {
    "train": TRAIN_PARQUET,
    "val": VAL_PARQUET,
    "stream": STREAM_PARQUET,
}

#: TransactionDT is a seconds offset from an unknown epoch, so absolute
#: wall-clock time is unrecoverable -- but hour-of-day and elapsed days are
#: both internally consistent and safe to derive.
SECONDS_PER_DAY: int = 86_400

# --------------------------------------------------------------------------
# Split boundaries, read from the manifest
# --------------------------------------------------------------------------

#: Tolerance when checking recorded cut points against the sixths formula.
#: Generous because the manifest may store rounded or int-cast seconds.
_BOUNDARY_TOLERANCE_SECONDS: float = 1.0


@dataclass(frozen=True)
class SplitBoundaries:
    """The TransactionDT cut points that produced the three parquet files.

    Derived in Phase 0 as::

        month3_end = dt_min + (dt_max - dt_min) / 6 * 3    # train | val
        month4_end = dt_min + (dt_max - dt_min) / 6 * 4    # val   | stream

    so that ``train`` is sixths 1-3, ``val`` is sixth 4, and ``stream`` is
    sixths 5-6.
    """

    dt_min: float
    dt_max: float
    month3_end: float
    month4_end: float

    @property
    def span_seconds(self) -> float:
        """Total TransactionDT range covered by the dataset."""
        return self.dt_max - self.dt_min

    @property
    def span_days(self) -> float:
        """Total range in days. IEEE-CIS is roughly 183."""
        return self.span_seconds / SECONDS_PER_DAY

    def day_offset(self, dt: float) -> float:
        """Convert an absolute TransactionDT to days since the first record."""
        return (dt - self.dt_min) / SECONDS_PER_DAY

    def split_of(self, dt: float) -> str:
        """Return which split a given TransactionDT belongs to.

        Boundaries are half-open (``dt < month3_end`` is train), matching
        how the Phase 0 masks were written.
        """
        if dt < self.month3_end:
            return "train"
        if dt < self.month4_end:
            return "val"
        return "stream"

    def verify(self) -> None:
        """Check the recorded cut points against the sixths formula.

        Guards against a manifest that was hand-edited, written by an older
        splitting rule, or paired with the wrong parquet files.
        """
        if self.span_seconds <= 0:
            raise ValueError(
                f"{SPLIT_MANIFEST_PATH.name}: dt_max ({self.dt_max}) must exceed "
                f"dt_min ({self.dt_min})."
            )
        for label, recorded, sixths in (
            ("month3_end", self.month3_end, 3),
            ("month4_end", self.month4_end, 4),
        ):
            expected = self.dt_min + self.span_seconds / 6 * sixths
            if not math.isclose(recorded, expected, abs_tol=_BOUNDARY_TOLERANCE_SECONDS):
                raise ValueError(
                    f"{SPLIT_MANIFEST_PATH.name}: {label} is {recorded}, but the "
                    f"sixths rule gives {expected} "
                    f"(dt_min + (dt_max - dt_min) / 6 * {sixths}). "
                    "The manifest and the parquet files may be out of step."
                )
        if not self.dt_min < self.month3_end < self.month4_end < self.dt_max:
            raise ValueError(
                f"{SPLIT_MANIFEST_PATH.name}: cut points are not strictly increasing "
                f"within [{self.dt_min}, {self.dt_max}]: "
                f"month3_end={self.month3_end}, month4_end={self.month4_end}."
            )


def _require(manifest: dict, key: str) -> float:
    """Pull a numeric field from the manifest with an actionable error."""
    if key not in manifest:
        raise KeyError(
            f"{SPLIT_MANIFEST_PATH}: missing required key {key!r}. "
            f"Keys present: {sorted(manifest)}"
        )
    value = manifest[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"{SPLIT_MANIFEST_PATH}: key {key!r} should be a number, "
            f"got {type(value).__name__} ({value!r})."
        )
    return float(value)


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    """Read and cache split_manifest.json.

    Deliberately lazy rather than loaded at import time: importing this
    module must keep working when the data root is missing, otherwise
    `python -m src.common.config` -- the tool you reach for when a Colab
    mount has gone wrong -- would fail with the very error you are trying
    to diagnose.
    """
    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {SPLIT_MANIFEST_PATH}. "
            "It is written by Phase 0 alongside the parquet files. "
            "If the data lives on Drive, set FRAUDSHIELD_DATA to that data root "
            "(currently resolving to "
            f"{DATA_ROOT})."
        )
    # utf-8-sig, not utf-8: a manifest written by a Windows tool (PowerShell's
    # Set-Content, Notepad) carries a UTF-8 BOM that makes strict utf-8 json.load
    # fail on character 0. utf-8-sig strips a BOM if present and is a no-op
    # otherwise, so the same file loads whether it was written on Colab or here.
    with SPLIT_MANIFEST_PATH.open(encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(
            f"{SPLIT_MANIFEST_PATH}: expected a JSON object at the top level, "
            f"got {type(manifest).__name__}."
        )
    return manifest


@lru_cache(maxsize=1)
def split_boundaries() -> SplitBoundaries:
    """Return the verified TransactionDT cut points from the manifest."""
    manifest = load_manifest()
    bounds = SplitBoundaries(
        dt_min=_require(manifest, "dt_min"),
        dt_max=_require(manifest, "dt_max"),
        month3_end=_require(manifest, "month3_end"),
        month4_end=_require(manifest, "month4_end"),
    )
    bounds.verify()
    return bounds


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
        ("VAL_PARQUET", VAL_PARQUET),
        ("STREAM_PARQUET", STREAM_PARQUET),
        ("SPLIT_MANIFEST_PATH", SPLIT_MANIFEST_PATH),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [f"FraudShield config  [{source}]", ""]
    lines += [f"  {name:<{width}}  {'x' if path.exists() else ' '}  {path}" for name, path in rows]
    lines.append("")
    lines.append("  (x = exists)")
    lines.append("")

    try:
        bounds = split_boundaries()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        lines.append(f"  split boundaries: UNAVAILABLE -- {exc}")
    else:
        lines.append(
            f"  split boundaries: span {bounds.span_days:.1f} days "
            f"({bounds.dt_min:.0f} -> {bounds.dt_max:.0f})"
        )
        lines.append(
            f"    train   dt <  {bounds.month3_end:.0f}  "
            f"(day 0 -> {bounds.day_offset(bounds.month3_end):.1f})"
        )
        lines.append(
            f"    val     dt <  {bounds.month4_end:.0f}  "
            f"(day {bounds.day_offset(bounds.month3_end):.1f} -> "
            f"{bounds.day_offset(bounds.month4_end):.1f})"
        )
        lines.append(
            f"    stream  dt >= {bounds.month4_end:.0f}  "
            f"(day {bounds.day_offset(bounds.month4_end):.1f} -> "
            f"{bounds.span_days:.1f})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
