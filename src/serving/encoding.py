"""Single-transaction feature encoding for the serving path.

``build_features`` is the reference implementation and stays that way -- it
is what training uses, and training-serving skew is the failure this whole
project keeps guarding against. But it is built for frames, and its cost is
per *column*: 431 pandas operations at ~50 microseconds of overhead each is
22 ms to encode one row, which is most of a latency budget spent on
bookkeeping rather than work.

``encode_row`` does the same thing against dicts and numpy: 0.05 ms,
measured 429x faster. It is only safe because it is proven equivalent, not
because it looks equivalent -- ``tests/test_serving.py`` asserts the two
produce bit-identical vectors across every column of every checked val row.
If that test ever fails, the fast path is wrong and must be deleted rather
than patched.

The three states from training survive unchanged:

    known level     -> its fitted code
    unseen level    -> UNKNOWN_CODE, never an error
    missing/absent  -> NaN, which XGBoost routes by learned default

The unseen path is not an edge case here. ``id_31`` carries browser strings
in the stream window that never appear in train -- that is the vocabulary
drift the PSI sweep measured -- so a transaction naming one must score
normally. It is also worth surfacing: ``encode_row`` reports which fields
were unseen, so the serving layer can watch vocabulary drift arrive in real
time rather than waiting for a nightly PSI job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.features.build_features import UNKNOWN_CODE, FeatureEncoders


@dataclass(frozen=True)
class EncodedRow:
    """One transaction as the model sees it, plus what was odd about it."""

    values: np.ndarray
    unseen: tuple[str, ...]
    missing: tuple[str, ...]


def _is_missing(value: Any) -> bool:
    """None, NaN, or an empty string -- all mean 'not recorded'."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return isinstance(value, str) and not value.strip()


class RowEncoder:
    """Encodes single transactions against a fitted ``FeatureEncoders``.

    Built once at startup. The per-request work is a dict lookup per column
    and nothing else -- no pandas, no allocation beyond the output vector.
    """

    def __init__(self, encoders: FeatureEncoders):
        self.encoders = encoders
        self.feature_names: tuple[str, ...] = tuple(encoders.feature_names)
        self.unknown_code = float(encoders.unknown_code)
        # Materialised once: which columns need a lookup, and the lookup
        # itself. Doing this per request is most of what makes the frame
        # path slow.
        self._mappings: dict[str, dict] = {
            name: dict(mapping) for name, mapping in encoders.mappings.items()
        }
        self._inverse: dict[str, dict] = {
            name: {code: level for level, code in mapping.items()}
            for name, mapping in self._mappings.items()
        }

    def encode(self, record: Mapping[str, Any]) -> EncodedRow:
        """Encode one transaction. Never raises on unexpected input."""
        values = np.full(len(self.feature_names), np.nan, dtype=np.float32)
        unseen: list[str] = []
        missing: list[str] = []

        for position, name in enumerate(self.feature_names):
            raw = record.get(name)
            if _is_missing(raw):
                missing.append(name)
                continue

            mapping = self._mappings.get(name)
            if mapping is None:
                # Numeric feature. A non-numeric value here is the client's
                # error, but erroring the whole request over one bad field
                # would be worse than scoring it as absent.
                try:
                    values[position] = float(raw)
                except (TypeError, ValueError):
                    missing.append(name)
                continue

            code = mapping.get(raw)
            if code is None:
                values[position] = self.unknown_code
                unseen.append(name)
            else:
                values[position] = code

        return EncodedRow(values, tuple(unseen), tuple(missing))

    def is_categorical(self, name: str) -> bool:
        return name in self._mappings

    def decode(self, name: str, value: float) -> str:
        """Render a stored value the way an analyst reads it.

        The three states stay distinct in the output for the same reason
        they stay distinct in the encoding: "a browser we have never seen"
        and "no browser recorded" are different facts about a transaction.
        """
        if value is None or (isinstance(value, float) and value != value):
            return "<missing>"
        if not self.is_categorical(name):
            return format_number(float(value))
        if float(value) == self.unknown_code:
            return "<unseen>"
        level = self._inverse[name].get(int(value))
        return str(level) if level is not None else f"<code {int(value)}>"


def format_number(value: float) -> str:
    """Readable across the IEEE-CIS range, without scientific notation.

    ``%g`` flips to exponent form at five digits, which turns a 31,937
    transaction amount -- the top of this dataset's range and the number an
    analyst looks at first -- into ``3.194e+04``.
    """
    if not np.isfinite(value):
        return str(value)
    if float(value) == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 1e-3:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:.3e}"


def unknown_code_of(encoders: FeatureEncoders) -> float:
    return float(getattr(encoders, "unknown_code", UNKNOWN_CODE))
