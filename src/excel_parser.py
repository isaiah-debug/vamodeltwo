"""
Excel → pipeline JSON converter.

Expected sheet layout (any column order, header row required):

    turn | speaker | utterance | [timestamp] | [code] | [notes]

Outputs a list of turn dicts that every downstream module reads:
    [
        {
            "turn": 1,
            "speaker": "Alice",
            "utterance": "I think we should lead with the budget.",
            "timestamp": "0:00:12",   # optional
            "code": "Directive",      # optional behavioral code
            "notes": ""               # optional
        },
        ...
    ]

Usage:
    from src.excel_parser import load_dialogue
    turns = load_dialogue("data/session_01.xlsx", sheet="Sheet1")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


# Column aliases — maps whatever the researcher typed to a canonical key.
_ALIASES: dict[str, str] = {
    "turn": "turn",
    "turn #": "turn",
    "turn_num": "turn",
    "#": "turn",
    "speaker": "speaker",
    "spk": "speaker",
    "participant": "speaker",
    "name": "speaker",
    "utterance": "utterance",
    "text": "utterance",
    "speech": "utterance",
    "content": "utterance",
    "dialogue": "utterance",
    "timestamp": "timestamp",
    "time": "timestamp",
    "start": "timestamp",
    "code": "code",
    "behavior": "code",
    "behavior_code": "code",
    "beh": "code",
    "notes": "notes",
    "note": "notes",
    "comments": "notes",
}

REQUIRED = {"speaker", "utterance"}


def load_dialogue(
    path: str | Path,
    sheet: str | int = 0,
    session_id: Optional[str] = None,
) -> list[dict]:
    """
    Read an Excel (.xlsx/.xls) or CSV file and return a list of turn dicts.

    Parameters
    ----------
    path      : Path to the .xlsx / .xls / .csv file.
    sheet     : Sheet name or zero-based index (Excel only, ignored for CSV).
    session_id: Optional label attached to every turn for multi-session merges.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, dtype=str, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(path, dtype=str, encoding="latin-1")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df.rename(columns=_ALIASES, inplace=True)
    else:
        # Auto-detect the header row: scan up to 20 rows for one containing
        # at least one recognised column name after aliasing.
        raw = pd.read_excel(path, sheet_name=sheet, dtype=str, header=None)
        header_row = 0
        for row_idx in range(min(20, len(raw))):
            cols = [str(v).strip().lower() for v in raw.iloc[row_idx]]
            aliased = {_ALIASES.get(c, c) for c in cols}
            if REQUIRED & aliased:
                header_row = row_idx
                break
        df = pd.read_excel(path, sheet_name=sheet, dtype=str, header=header_row)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df.rename(columns=_ALIASES, inplace=True)

    df.dropna(how="all", inplace=True)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Sheet '{sheet}' is missing required columns: {missing}.\n"
            f"Found columns: {list(df.columns)}\n"
            "Rename them or add them to the _ALIASES table in excel_parser.py."
        )

    # Normalise
    df["speaker"] = df["speaker"].str.strip()
    df["utterance"] = df["utterance"].fillna("").str.strip()
    df = df[df["utterance"] != ""]          # drop blank-utterance rows
    # Drop template hint rows: speaker values containing "/" are placeholders
    # like "A/B/C/D" — real speaker values are single names/IDs
    df = df[~df["speaker"].str.contains("/", na=False)]

    df = df.reset_index(drop=True)
    if "turn" not in df.columns:
        df.insert(0, "turn", range(1, len(df) + 1))
    else:
        numeric = pd.to_numeric(df["turn"], errors="coerce")
        fallback = pd.Series(range(1, len(df) + 1), index=df.index)
        df["turn"] = numeric.where(numeric.notna(), fallback).astype(int)

    for opt in ("timestamp", "code", "notes"):
        if opt not in df.columns:
            df[opt] = ""

    if session_id:
        df["session"] = session_id

    records = df[
        ["turn", "speaker", "utterance", "timestamp", "code", "notes"]
        + (["session"] if session_id else [])
    ].to_dict(orient="records")

    # Convert numpy ints → plain Python ints for JSON serialisability
    for r in records:
        r["turn"] = int(r["turn"])
        for k, v in r.items():
            if hasattr(v, "item"):
                r[k] = v.item()

    return records


def save_json(turns: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2, ensure_ascii=False)
    print(f"[excel_parser] saved {len(turns)} turns → {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.excel_parser <path/to/file.xlsx> [sheet_name]")
        sys.exit(1)
    sheet_arg = sys.argv[2] if len(sys.argv) > 2 else 0
    turns = load_dialogue(sys.argv[1], sheet=sheet_arg)
    print(f"Loaded {len(turns)} turns from {sys.argv[1]}")
    for t in turns[:5]:
        print(f"  {t['turn']:>3}  {t['speaker']:<15}  {t['utterance'][:60]}")
