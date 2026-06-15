#!/usr/bin/env python3
"""Standalone pilot transcription script for Laguna.

Copy this file into:
    /project/SZhou_1896/Pilot Test Apr 2026

Then run it from that folder:
    python3 standalone_transcribe_pilot.py

It transcribes the four participant-labeled MP4 files:
    A (Pink M).mp4
    B (Grey M).mp4
    C (Grey F).mp4
    D (Brown F).mp4

Output:
    output/pilot_transcription_test/utterances.csv

This script intentionally has no imports from the git repository. It only
requires system ffmpeg plus Python packages for WhisperX/PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path


DEFAULT_SPEAKER_FILES = {
    "A": "A (Pink M).mp4",
    "B": "B (Grey M).mp4",
    "C": "C (Grey F).mp4",
    "D": "D (Brown F).mp4",
}

CSV_COLUMNS = (
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

BROADCAST_RE = re.compile(
    r"\b(everyone|everybody|all of you|you all|y'all|you guys|guys|team|folks)\b",
    flags=re.IGNORECASE,
)
SECOND_PERSON_RE = re.compile(
    r"\b(you|your|yours|you're|youre|you'll|youll|you've|youve)\b",
    flags=re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="pilot_transcription_test")
    parser.add_argument("--media-dir", default=".", help="Folder containing the four participant MP4s")
    parser.add_argument("--out", default="", help="Output CSV path")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--language", default="en")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--audio-track", type=int, default=0)
    parser.add_argument(
        "--speaker-file",
        action="append",
        default=[],
        metavar="PLAYER=FILENAME",
        help="Override/add a speaker MP4 mapping, e.g. A='A (Pink M).mp4'",
    )
    parser.add_argument(
        "--player-name",
        action="append",
        default=[],
        metavar="PLAYER=NAME",
        help="Participant names for name-mention addressee inference",
    )
    args = parser.parse_args()

    media_dir = Path(args.media_dir).expanduser().resolve()
    out_csv = Path(args.out) if args.out else Path("output") / args.session / "utterances.csv"
    out_csv = out_csv.expanduser().resolve()
    work_dir = out_csv.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    speaker_files = dict(DEFAULT_SPEAKER_FILES)
    speaker_files.update(parse_pairs(args.speaker_file))
    player_aliases = {player: [name] for player, name in parse_pairs(args.player_name).items()}

    print("Media directory:", media_dir)
    print("Output CSV:", out_csv)
    print("Speaker files:")
    for speaker, filename in speaker_files.items():
        path = media_dir / filename
        print(f"  {speaker}: {path} exists={path.exists()}")
        if not path.exists():
            raise FileNotFoundError(path)

    wx, model, align_model, metadata, device = load_whisperx(
        model_name=args.model,
        language=args.language,
        compute_type=args.compute_type,
    )

    all_turns = []
    for speaker, filename in sorted(speaker_files.items()):
        mp4_path = media_dir / filename
        speaker_work = work_dir / speaker
        speaker_work.mkdir(parents=True, exist_ok=True)
        wav_path = extract_audio_track(
            mp4_path=mp4_path,
            out_dir=speaker_work,
            track_index=args.audio_track,
        )
        all_turns.extend(
            transcribe_wav(
                wx=wx,
                model=model,
                align_model=align_model,
                metadata=metadata,
                device=device,
                wav_path=wav_path,
                speaker=speaker,
                source_file=filename,
                audio_track=args.audio_track,
                session=args.session,
                batch_size=args.batch_size,
            )
        )

    all_turns.sort(key=lambda turn: (float(turn["start_s"]), str(turn["speaker"])))
    for index, turn in enumerate(all_turns, start=1):
        turn["turn"] = index

    enriched = infer_addressees(
        all_turns,
        players=sorted(speaker_files),
        player_aliases=player_aliases,
    )
    write_csv(enriched, out_csv)

    turns_json = out_csv.parent / "turns.json"
    with turns_json.open("w", encoding="utf-8") as handle:
        json.dump(enriched, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {len(enriched)} turns to {out_csv}")
    print(f"Cached JSON: {turns_json}")
    return 0


def parse_pairs(items: list[str]) -> dict[str, str]:
    pairs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def load_whisperx(model_name: str, language: str, compute_type: str):
    try:
        import torch
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "Install transcription dependencies first: python3 -m pip install torch whisperx"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading WhisperX model {model_name} on {device}")
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    return whisperx, model, align_model, metadata, device


def extract_audio_track(mp4_path: Path, out_dir: Path, track_index: int) -> Path:
    wav_path = out_dir / f"{mp4_path.stem}_track{track_index}.wav"
    if wav_path.exists() and wav_path.stat().st_mtime > mp4_path.stat().st_mtime:
        print(f"Using cached WAV: {wav_path}")
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-map",
        f"0:a:{track_index}",
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    print("Extracting audio:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {mp4_path}:\n{result.stderr}")
    return wav_path


def transcribe_wav(
    wx,
    model,
    align_model,
    metadata,
    device: str,
    wav_path: Path,
    speaker: str,
    source_file: str,
    audio_track: int,
    session: str,
    batch_size: int,
) -> list[dict]:
    cache = wav_path.with_name(f"{wav_path.stem}_turns.json")
    if cache.exists():
        print(f"Using cached turns: {cache}")
        with cache.open(encoding="utf-8") as handle:
            return json.load(handle)

    print(f"Transcribing {wav_path.name} as speaker {speaker}")
    audio = wx.load_audio(str(wav_path))
    result = model.transcribe(audio, batch_size=batch_size)
    result = wx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    turns = []
    for segment in result.get("segments", []):
        utterance = str(segment.get("text", "")).strip()
        if not utterance:
            continue
        start_s = float(segment.get("start", 0.0) or 0.0)
        end_s = float(segment.get("end", start_s) or start_s)
        turns.append(
            {
                "session": session,
                "source_file": source_file,
                "audio_track": audio_track,
                "start_s": start_s,
                "end_s": end_s,
                "local_start_s": start_s,
                "local_end_s": end_s,
                "timestamp": seconds_to_timestamp(start_s),
                "speaker": speaker,
                "utterance": utterance,
            }
        )

    with cache.open("w", encoding="utf-8") as handle:
        json.dump(turns, handle, indent=2, ensure_ascii=False)
    return turns


def infer_addressees(turns: list[dict], players: list[str], player_aliases: dict[str, list[str]]) -> list[dict]:
    alias_patterns = build_alias_patterns(players, player_aliases)
    enriched = []
    for index, turn in enumerate(turns):
        out = dict(turn)
        speaker = out["speaker"]
        text = out.get("utterance", "")

        addressee = ""
        method = "unknown"
        if BROADCAST_RE.search(text):
            addressee = "All"
            method = "broadcast_keyword"
        else:
            mentioned = [
                player for player in players
                if player != speaker and any(pattern.search(text) for pattern in alias_patterns[player])
            ]
            if mentioned:
                addressee = ";".join(mentioned)
                method = "name_mention"
            elif SECOND_PERSON_RE.search(text):
                addressee = nearby_speaker(turns, index, speaker, prefer_previous=True)
                method = "pronoun_context" if addressee else "unknown"
            if not addressee:
                addressee = nearby_speaker(turns, index, speaker, prefer_previous=False)
                method = "sequential_context" if addressee else method

        out["addressee"] = addressee
        out["addressee_method"] = method
        enriched.append(out)
    return enriched


def build_alias_patterns(players: list[str], player_aliases: dict[str, list[str]]):
    patterns = {}
    for player in players:
        aliases = list(player_aliases.get(player, []))
        aliases.extend([f"player {player}", f"participant {player}", f"person {player}"])
        patterns[player] = [
            re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE)
            for alias in aliases
            if alias
        ]
    return patterns


def nearby_speaker(turns: list[dict], index: int, speaker: str, prefer_previous: bool) -> str:
    directions = (-1, 1) if prefer_previous else (1, -1)
    for direction in directions:
        cursor = index + direction
        while 0 <= cursor < len(turns):
            candidate = turns[cursor].get("speaker", "")
            if candidate and candidate != speaker:
                return candidate
            cursor += direction
    return ""


def write_csv(turns: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for turn in turns:
            writer.writerow({column: csv_value(turn.get(column, "")) for column in CSV_COLUMNS})


def csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def seconds_to_timestamp(seconds: float) -> str:
    total = int(math.floor(float(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    remainder = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
