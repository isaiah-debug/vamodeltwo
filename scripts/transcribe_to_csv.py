"""Generate an utterance CSV from experiment media or coded transcripts.

Examples
--------
Headcam or audio files:
    python scripts/transcribe_to_csv.py \
        --media data/videos/cam_A.mp4 data/videos/cam_B.mp4 \
        --camera-map cam_A.mp4=A cam_B.mp4=B \
        --player-name A=Jordan B=Elis \
        --out output/session_01/utterances.csv

Existing coded CSV directory:
    python scripts/transcribe_to_csv.py --from-csv data --out output/session_01/utterances.csv

Existing turn JSON:
    python scripts/transcribe_to_csv.py --from-json output/session_01/video_turns.json
"""

from __future__ import annotations

import argparse
import json
import os
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", help="Project YAML config with videos and players")
    inputs.add_argument("--media", "--videos", nargs="+", help="Video/audio files")
    inputs.add_argument("--from-csv", help="Directory of per-player coded CSV files")
    inputs.add_argument("--from-json", help="Existing video_turns.json or turn list JSON")

    parser.add_argument("--camera-map", nargs="+", metavar="FILE=PLAYER")
    parser.add_argument("--player-name", nargs="+", metavar="PLAYER=NAME")
    parser.add_argument("--players", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--session", default="session_01")
    parser.add_argument("--out", help="Output CSV path")
    parser.add_argument("--model", default="large-v3-turbo", help="WhisperX model name")
    parser.add_argument("--language", default="en")
    parser.add_argument("--hf-token", default="", help="HuggingFace token for diarization")
    parser.add_argument("--no-face", action="store_true", help="Skip face/video ID mapping")
    parser.add_argument(
        "--no-sequential-addressee",
        action="store_true",
        help="Leave addressee unknown if no coded/name/pronoun cue is found",
    )
    args = parser.parse_args(argv)

    turns, players, player_aliases, session_id, out_path = _load_turns(args)

    from src.addressee_inference import infer_addressees
    from src.export_utterances import export_utterances_csv

    enriched = infer_addressees(
        turns,
        players=players,
        player_aliases=player_aliases,
        use_sequential=not args.no_sequential_addressee,
    )
    destination = export_utterances_csv(enriched, out_path)

    print(f"[transcribe_to_csv] wrote {len(enriched)} utterances -> {destination}")
    return 0


def _load_turns(args) -> tuple[list[dict], list[str], dict[str, list[str]], str, Path]:
    if args.config:
        return _load_from_config(Path(args.config), args)
    if args.from_csv:
        return _load_from_csv(Path(args.from_csv), args)
    if args.from_json:
        return _load_from_json(Path(args.from_json), args)
    return _load_from_media([Path(path) for path in args.media], args)


def _load_from_config(path: Path, args) -> tuple[list[dict], list[str], dict[str, list[str]], str, Path]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    session_id = cfg["project"]["session_id"]
    players = [str(player["label"]) for player in cfg.get("players", [])]
    aliases = {
        str(player["label"]): [str(player["name"])]
        for player in cfg.get("players", [])
        if player.get("name")
    }

    data_dir = Path(cfg["videos"].get("data_dir", "."))
    media_paths = []
    camera_to_player = {}
    for headcam in cfg["videos"].get("headcams", []):
        media_paths.append(data_dir / headcam["file"])
        camera_to_player[headcam["file"]] = headcam["player"]
    for camera in cfg["videos"].get("security_cameras", []):
        media_paths.append(data_dir / camera["file"])

    transcription = cfg.get("transcription", {})
    face_id = cfg.get("face_id", {})
    out_dir = Path(cfg.get("output", {}).get("dir", "output")) / session_id
    out_path = Path(args.out) if args.out else out_dir / "utterances.csv"

    turns = _process_media(
        media_paths=media_paths,
        out_dir=out_dir,
        session_id=session_id,
        camera_to_player=camera_to_player or None,
        players=players,
        model=transcription.get("whisper_model", args.model),
        language=transcription.get("language", args.language),
        min_speakers=int(transcription.get("min_speakers", 2)),
        max_speakers=int(transcription.get("max_speakers", len(players) or 4)),
        hf_token=args.hf_token or os.environ.get("HF_TOKEN", ""),
        run_face_id=bool(face_id.get("enabled", True)),
        sample_video_every_n=int(face_id.get("sample_every_n_frames", 15)),
    )
    return turns, players, aliases, session_id, out_path


def _load_from_csv(path: Path, args) -> tuple[list[dict], list[str], dict[str, list[str]], str, Path]:
    from src.multi_csv_loader import load_experiment

    session_id = args.session
    turns, _edges = load_experiment(path, session_id=session_id)
    players = args.players
    aliases = _aliases_from_args(args.player_name)
    out_path = Path(args.out) if args.out else Path("output") / session_id / "utterances.csv"
    return turns, players, aliases, session_id, out_path


def _load_from_json(path: Path, args) -> tuple[list[dict], list[str], dict[str, list[str]], str, Path]:
    with path.open(encoding="utf-8") as handle:
        turns = json.load(handle)
    if not isinstance(turns, list):
        raise ValueError(f"Expected a list of turns in {path}")

    session_id = args.session
    players = args.players
    aliases = _aliases_from_args(args.player_name)
    out_path = Path(args.out) if args.out else path.with_name("utterances.csv")
    return turns, players, aliases, session_id, out_path


def _load_from_media(
    media_paths: list[Path],
    args,
) -> tuple[list[dict], list[str], dict[str, list[str]], str, Path]:
    session_id = args.session
    players = args.players
    aliases = _aliases_from_args(args.player_name)
    out_dir = Path("output") / session_id
    out_path = Path(args.out) if args.out else out_dir / "utterances.csv"

    turns = _process_media(
        media_paths=media_paths,
        out_dir=out_dir,
        session_id=session_id,
        camera_to_player=parse_assignments(args.camera_map) or None,
        players=players,
        model=args.model,
        language=args.language,
        min_speakers=2,
        max_speakers=len(players),
        hf_token=args.hf_token or os.environ.get("HF_TOKEN", ""),
        run_face_id=not args.no_face,
        sample_video_every_n=15,
    )
    return turns, players, aliases, session_id, out_path


def _process_media(
    media_paths: list[Path],
    out_dir: Path,
    session_id: str,
    camera_to_player: dict[str, str] | None,
    players: list[str],
    model: str,
    language: str,
    min_speakers: int,
    max_speakers: int,
    hf_token: str,
    run_face_id: bool,
    sample_video_every_n: int,
) -> list[dict]:
    missing = [str(path) for path in media_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Media file(s) not found: {', '.join(missing)}")

    from src.video_pipeline import process_session

    return process_session(
        video_paths=media_paths,
        out_dir=out_dir,
        session_id=session_id,
        camera_to_player=camera_to_player,
        player_labels=players,
        whisper_model=model,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        run_face_id=run_face_id,
        sample_video_every_n=sample_video_every_n,
    )


def _aliases_from_args(items: list[str] | None) -> dict[str, list[str]]:
    return {key: [value] for key, value in parse_assignments(items).items()}


if __name__ == "__main__":
    raise SystemExit(main())
