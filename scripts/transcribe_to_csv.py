"""Transcribe seven multi-track MP4 files into one dialogue CSV.

Expected experiment input:
  - 7 MP4 files
  - each MP4 has 4 isolated audio tracks
  - each audio track corresponds to one participant/speaker

Example:
    python scripts/transcribe_to_csv.py \
        --media data/videos/session_part1.mp4 data/videos/session_part2.mp4 \
        --track-map 0=A 1=B 2=C 3=D \
        --player-name A=Jordan B=Elis C=Anna D=Isaiah \
        --session session_01 \
        --out output/session_01/utterances.csv

Use --file-offset when MP4 files are sequential chunks and timestamps should be
global across the full session:
    --file-offset session_part2.mp4=1800 session_part3.mp4=3600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_assignments(items: list[str] | None) -> dict[str, str]:
    """Parse KEY=VALUE CLI pairs."""
    assignments: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {item}")
        assignments[key] = value
    return assignments


def parse_track_map(items: list[str] | None, players: list[str]) -> dict[int, str]:
    """Parse audio track to speaker mapping."""
    if not items:
        return {index: player for index, player in enumerate(players)}

    raw = parse_assignments(items)
    track_map: dict[int, str] = {}
    for key, value in raw.items():
        try:
            track_index = int(key)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Track index must be an integer: {key}") from exc
        track_map[track_index] = value
    return track_map


def parse_offsets(items: list[str] | None) -> dict[str, float]:
    """Parse FILE=SECONDS offsets for sequential MP4 chunks."""
    raw = parse_assignments(items)
    offsets: dict[str, float] = {}
    for key, value in raw.items():
        try:
            offsets[key] = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Offset must be seconds, got {key}={value}") from exc
    return offsets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", help="Project YAML config")
    inputs.add_argument("--media", nargs="+", help="Seven MP4 files to process")
    inputs.add_argument("--from-json", help="Existing cached turns JSON")

    parser.add_argument("--track-map", nargs="+", metavar="TRACK=PLAYER")
    parser.add_argument("--file-offset", nargs="+", metavar="FILE=SECONDS")
    parser.add_argument("--player-name", nargs="+", metavar="PLAYER=NAME")
    parser.add_argument("--players", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--expected-files", type=int, default=7)
    parser.add_argument("--session", default="session_01")
    parser.add_argument("--out", help="Output CSV path")
    parser.add_argument("--model", default="large-v3-turbo", help="WhisperX model name")
    parser.add_argument("--language", default="en")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument(
        "--no-sequential-addressee",
        action="store_true",
        help="Leave addressee unknown if no coded/name/pronoun cue is found",
    )
    args = parser.parse_args(argv)

    turns, players, aliases, out_path = _load_turns(args)

    from src.addressee_inference import infer_addressees
    from src.export_utterances import export_utterances_csv

    enriched = infer_addressees(
        turns,
        players=players,
        player_aliases=aliases,
        use_sequential=not args.no_sequential_addressee,
    )
    destination = export_utterances_csv(enriched, out_path)
    print(f"[transcribe_to_csv] wrote {len(enriched)} utterances -> {destination}")
    return 0


def _load_turns(args) -> tuple[list[dict], list[str], dict[str, list[str]], Path]:
    if args.config:
        return _load_from_config(Path(args.config), args)
    if args.from_json:
        return _load_from_json(Path(args.from_json), args)
    return _load_from_media([Path(path) for path in args.media], args)


def _load_from_config(path: Path, args) -> tuple[list[dict], list[str], dict[str, list[str]], Path]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    session_id = cfg["project"]["session_id"]
    players = [str(player["label"]) for player in cfg.get("players", [])]
    aliases = {
        str(player["label"]): [str(player["name"])]
        for player in cfg.get("players", [])
        if player.get("name")
    }
    video_cfg = cfg.get("videos", {})
    data_dir = Path(video_cfg.get("data_dir", "."))
    media_paths = [data_dir / item["file"] for item in video_cfg.get("files", [])]
    offsets = {
        item["file"]: float(item.get("offset_s", 0.0))
        for item in video_cfg.get("files", [])
    }
    track_map = {
        int(track["index"]): str(track["player"])
        for track in video_cfg.get("audio_tracks", [])
    } or parse_track_map(None, players)

    transcription = cfg.get("transcription", {})
    out_dir = Path(cfg.get("output", {}).get("dir", "output")) / session_id
    out_path = Path(args.out) if args.out else out_dir / "utterances.csv"

    turns = _process_media(
        media_paths=media_paths,
        track_map=track_map,
        file_offsets=offsets,
        out_dir=out_dir,
        session_id=session_id,
        expected_files=int(video_cfg.get("expected_files", args.expected_files)),
        model=transcription.get("whisper_model", args.model),
        language=transcription.get("language", args.language),
        batch_size=int(transcription.get("batch_size", args.batch_size)),
        compute_type=transcription.get("compute_type", args.compute_type),
    )
    return turns, players, aliases, out_path


def _load_from_json(path: Path, args) -> tuple[list[dict], list[str], dict[str, list[str]], Path]:
    with path.open(encoding="utf-8") as handle:
        turns = json.load(handle)
    if not isinstance(turns, list):
        raise ValueError(f"Expected a list of turns in {path}")

    aliases = _aliases_from_args(args.player_name)
    out_path = Path(args.out) if args.out else path.with_name("utterances.csv")
    return turns, args.players, aliases, out_path


def _load_from_media(
    media_paths: list[Path],
    args,
) -> tuple[list[dict], list[str], dict[str, list[str]], Path]:
    players = args.players
    aliases = _aliases_from_args(args.player_name)
    track_map = parse_track_map(args.track_map, players)
    offsets = parse_offsets(args.file_offset)
    out_dir = Path("output") / args.session
    out_path = Path(args.out) if args.out else out_dir / "utterances.csv"

    turns = _process_media(
        media_paths=media_paths,
        track_map=track_map,
        file_offsets=offsets,
        out_dir=out_dir,
        session_id=args.session,
        expected_files=args.expected_files,
        model=args.model,
        language=args.language,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
    )
    return turns, players, aliases, out_path


def _process_media(
    media_paths: list[Path],
    track_map: dict[int, str],
    file_offsets: dict[str, float],
    out_dir: Path,
    session_id: str,
    expected_files: int,
    model: str,
    language: str,
    batch_size: int,
    compute_type: str,
) -> list[dict]:
    if expected_files and len(media_paths) != expected_files:
        raise ValueError(f"Expected {expected_files} MP4 files, got {len(media_paths)}")
    if len(track_map) != 4:
        raise ValueError(f"Expected 4 speaker audio tracks, got {len(track_map)}")

    missing = [str(path) for path in media_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Media file(s) not found: {', '.join(missing)}")

    from src.video_pipeline import process_multitrack_session

    return process_multitrack_session(
        media_paths=media_paths,
        track_map=track_map,
        out_dir=out_dir,
        session_id=session_id,
        file_offsets=file_offsets,
        whisper_model=model,
        language=language,
        batch_size=batch_size,
        compute_type=compute_type,
    )


def _aliases_from_args(items: list[str] | None) -> dict[str, list[str]]:
    return {key: [value] for key, value in parse_assignments(items).items()}


if __name__ == "__main__":
    raise SystemExit(main())
