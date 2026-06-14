"""
Multi-player CSV loader for the escape room experiment.

Handles:
  - Forward-filled speaker column (blank rows = same speaker as above)
  - Explicit receiver column (A / B / C / D / All / B and C / A and B / self / None / Other)
  - Mixed timestamp formats: M:SS, MM:SS, H:MM:SS, MMSS (no colon)
  - Per-player CSV files merged into a single sorted timeline
  - Builds a DIRECTED graph from explicit speaker→receiver pairs (more accurate than
    sequential adjacency inference)

Usage:
    from src.multi_csv_loader import load_experiment
    turns, edges = load_experiment("data/", player_files={...})
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

CORE_PLAYERS = {"A", "B", "C", "D"}

# Receiver values that represent external / unknown targets — no graph edge
NON_PLAYER_RECEIVERS = {
    "self", "none", "other", "zhou", "owen", "prof",
    "the room", "room", "self/the room",
}


def parse_timestamp(ts: str) -> Optional[float]:
    """
    Convert a timestamp string to total seconds.

    Formats handled:
      "0:35"         -> 35.0   (M:SS)
      "2:14"         -> 134.0  (M:SS)
      "10:51"        -> 651.0  (MM:SS)
      "0:01:00"      -> 60.0   (H:MM:SS where H=0, MM=1 minute)
      "24:17:00"     -> 1457.0 (H:MM:SS-style but really MM:SS with :00 suffix)
      "0056"         -> 56.0   (MMSS without colons)
      "0105"         -> 65.0   (MMSS without colons = 1min 5sec)
    """
    if not ts or pd.isna(ts):
        return None
    ts = str(ts).strip()
    if not ts:
        return None

    # Remove trailing whitespace
    ts = ts.strip()

    # MMSS without colon (4 digits)
    if re.fullmatch(r"\d{4}", ts):
        mm, ss = int(ts[:2]), int(ts[2:])
        return mm * 60 + ss

    parts = ts.split(":")
    try:
        if len(parts) == 2:
            # M:SS or MM:SS
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            # Could be H:MM:SS or — common in this dataset — MM:SS:00
            # Heuristic: if the first part > 23, treat as MM:SS (ignore last part)
            h_or_mm = int(parts[0])
            mm_or_ss = int(parts[1])
            ss_or_extra = float(parts[2])
            if h_or_mm > 23:
                # MM:SS:00 format — the :00 is spurious Excel formatting
                return h_or_mm * 60 + mm_or_ss
            else:
                # True H:MM:SS
                return h_or_mm * 3600 + mm_or_ss * 60 + ss_or_extra
    except (ValueError, TypeError):
        return None
    return None


def parse_receivers(rec_str: str, all_players: set[str] = CORE_PLAYERS) -> list[str]:
    """
    Parse a receiver cell into a list of player IDs that receive this utterance.

    Examples:
      "A"        -> ["A"]
      "B and C"  -> ["B", "C"]
      "A and B"  -> ["A", "B"]
      "A/B"      -> ["A", "B"]
      "All"      -> all players except the sender (resolved by caller)
      "self"     -> []
      "Other"    -> []
      "Zhou"     -> []
      "B?"       -> ["B"]  (coder uncertainty marker stripped)
    """
    if not rec_str or pd.isna(rec_str):
        return []

    raw = str(rec_str).strip()
    low = raw.lower()

    if low in NON_PLAYER_RECEIVERS:
        return []
    if low == "all":
        return ["ALL"]  # caller expands to all other players

    # Strip uncertainty markers like "B?" or "B?"
    raw = re.sub(r"\?$", "", raw).strip()

    # Split on "and", "/", "," — case-insensitive
    parts = re.split(r"\s+and\s+|[/,]", raw, flags=re.IGNORECASE)
    result = []
    for p in parts:
        p = p.strip().upper()
        if p in {pl.upper() for pl in all_players}:
            result.append(p)
    return result


def load_player_csv(path: Path, player_id: str) -> pd.DataFrame:
    """
    Load one player's CSV file, forward-filling blank speaker cells.
    Returns a clean DataFrame with columns:
        speaker, receiver_raw, start_s, end_s, utterance, notes, player_file
    """
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename common variants
    col_map = {
        "transcript": "utterance", "speech": "utterance", "text": "utterance",
        "notes_challenges": "notes", "notes": "notes",
        "receiver": "receiver_raw",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    for col in ("utterance", "receiver_raw", "notes"):
        if col not in df.columns:
            df[col] = ""

    # Forward-fill speaker: blank = same as previous speaker
    df["speaker"] = df["speaker"].replace("", pd.NA).ffill()
    df["speaker"] = df["speaker"].fillna(player_id).str.strip().str.upper()

    # Only keep rows for this player (in case of multi-speaker files)
    df = df[df["speaker"] == player_id.upper()].copy()

    # Parse timestamps
    df["start_s"] = df["start"].apply(parse_timestamp)
    df["end_s"]   = df["end"].apply(parse_timestamp)

    df["utterance"]    = df["utterance"].fillna("").str.strip()
    df["receiver_raw"] = df["receiver_raw"].fillna("").str.strip()
    df["notes"]        = df["notes"].fillna("").str.strip()
    df["player_file"]  = player_id

    df = df[df["utterance"] != ""]
    df = df[df["start_s"].notna()].copy()

    return df[["speaker", "receiver_raw", "start_s", "end_s", "utterance", "notes", "player_file"]]


def load_experiment(
    data_dir: str | Path,
    player_files: Optional[dict[str, str]] = None,
    session_id: str = "escape_room_01",
) -> tuple[list[dict], list[dict]]:
    """
    Load all player CSV files and return:
      turns : list of turn dicts (sorted by start_s, deduplicated)
      edges : list of explicit directed edge dicts {from, to, weight, utterances}

    Parameters
    ----------
    data_dir     : Directory containing the CSV files.
    player_files : Dict mapping player ID -> filename. Defaults to auto-detect
                   files matching player_*.csv or *_A_*.csv patterns.
    session_id   : Label added to every turn dict.
    """
    data_dir = Path(data_dir)

    if player_files is None:
        # Auto-detect
        player_files = {}
        for p in ("A", "B", "C", "D"):
            matches = list(data_dir.glob(f"*player_{p}*.csv")) + \
                      list(data_dir.glob(f"*_{p}_*.csv")) + \
                      list(data_dir.glob(f"player_{p}*.csv"))
            if matches:
                player_files[p] = matches[0].name

    dfs = []
    for pid, fname in player_files.items():
        fpath = data_dir / fname
        if fpath.exists():
            df = load_player_csv(fpath, pid)
            dfs.append(df)
            print(f"  [loader] Player {pid}: {len(df)} turns from {fname}")
        else:
            print(f"  [loader] WARNING: {fpath} not found, skipping Player {pid}")

    if not dfs:
        raise FileNotFoundError(f"No player CSV files found in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True).sort_values("start_s").reset_index(drop=True)

    all_players = set(combined["speaker"].unique())

    # Build turn dicts
    turns = []
    for i, row in combined.iterrows():
        turns.append({
            "turn":        i + 1,
            "speaker":     row["speaker"],
            "utterance":   row["utterance"],
            "timestamp":   f"{int(row['start_s'] // 60):02d}:{int(row['start_s'] % 60):02d}",
            "start_s":     row["start_s"],
            "end_s":       row["end_s"],
            "receiver_raw": row["receiver_raw"],
            "notes":       row["notes"],
            "session":     session_id,
        })

    # Build explicit directed edges from receiver column
    from collections import defaultdict
    edge_counts: dict[tuple[str, str], list[str]] = defaultdict(list)

    for turn in turns:
        src = turn["speaker"]
        rec_raw = turn["receiver_raw"]
        receivers = parse_receivers(rec_raw, all_players)

        # Expand "ALL" to all other players
        if "ALL" in receivers:
            receivers = [p for p in all_players if p != src]

        for dst in receivers:
            if dst != src and dst in all_players:
                edge_counts[(src, dst)].append(turn["utterance"])

    edges = [
        {"from": src, "to": dst, "weight": len(utts), "utterances": utts}
        for (src, dst), utts in edge_counts.items()
    ]

    print(f"  [loader] Total: {len(turns)} turns, {len(edges)} directed edges")
    return turns, edges


def build_explicit_graph(edges: list[dict]):
    """Build a networkx DiGraph from the explicit edge list."""
    import networkx as nx
    G = nx.DiGraph()
    for e in edges:
        if G.has_edge(e["from"], e["to"]):
            G[e["from"]][e["to"]]["weight"] += e["weight"]
        else:
            G.add_edge(e["from"], e["to"], weight=e["weight"])
    return G
