"""CSV export helpers for transcript utterances."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


UTTERANCE_COLUMNS = (
    "session",
    "turn",
    "source_file",
    "audio_track",
    "start_s",
    "end_s",
    "local_start_s",
    "local_end_s",
    "timestamp",
    "speaker",
    "addressee",
    "utterance",
    "addressee_method",
)


def export_utterances_csv(
    turns: Sequence[Mapping],
    out_path: str | Path,
    columns: Sequence[str] = UTTERANCE_COLUMNS,
) -> Path:
    """Write utterance-level transcript rows to CSV.

    The default schema is intentionally narrow: one row per utterance with
    timing, speaker, inferred/coded addressee, text, and inference provenance.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for turn in turns:
            row = {column: _csv_value(turn.get(column, "")) for column in columns}
            if not row.get("timestamp"):
                row["timestamp"] = _timestamp_from_seconds(turn.get("start_s"))
            writer.writerow(row)

    return out_path


def _csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _timestamp_from_seconds(value) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes:02d}:{remainder:02d}"
