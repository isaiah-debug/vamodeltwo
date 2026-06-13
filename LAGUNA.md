#!/usr/bin/env python3
"""Process ONE session through ONE pipeline stage.

This is the architectural unit of the whole repo: notebooks, teammates,
and the Slurm job array all call this same script. Keep it boring.

Usage:
    python scripts/process_one_session.py --session s01_groupA --stage audio
    python scripts/process_one_session.py --session s01_groupA --stage features
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def stage_audio(session: str, cfg: dict) -> None:
    """Transcribe + diarize via WhisperX. Requires the audio venv."""
    data_dir = REPO_ROOT / cfg["paths"]["data_dir"]
    out_dir = REPO_ROOT / cfg["paths"]["output_dir"] / session
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = data_dir / f"{session}.wav"
    if not wav.exists():
        sys.exit(f"Not found: {wav}\nNaming convention: see data/README.md")

    hf_token = os.environ.get("HF_TOKEN", "")
    if cfg["audio"]["diarize"] and not hf_token:
        sys.exit("HF_TOKEN not set. Copy .env.example to .env, fill it, and "
                 "run:  export $(grep -v '^#' .env | xargs)")

    cmd = [
        "whisperx", str(wav),
        "--model", cfg["audio"]["whisper_model"],
        "--compute_type", cfg["audio"]["compute_type"],
        "--language", cfg["audio"]["language"],
        "--output_dir", str(out_dir),
        "--output_format", "json",
    ]
    if cfg["audio"]["diarize"]:
        cmd += ["--diarize", "--hf_token", hf_token,
                "--min_speakers", str(cfg["audio"]["min_speakers"]),
                "--max_speakers", str(cfg["audio"]["max_speakers"])]

    print(f"[audio] {session}: running WhisperX...")
    subprocess.run(cmd, check=True)
    print(f"[audio] done -> {out_dir}")


def stage_features(session: str, cfg: dict) -> None:
    """Compute turn-taking features from the transcript JSON."""
    from pipeline.features.turns import compute_speaker_features

    out_dir = REPO_ROOT / cfg["paths"]["output_dir"] / session
    transcripts = list(out_dir.glob("*.json"))
    if not transcripts:
        sys.exit(f"No transcript JSON in {out_dir} — run --stage audio first.")

    with open(transcripts[0]) as f:
        transcript = json.load(f)

    features = compute_speaker_features(
        transcript,
        overlap_ms=cfg["features"]["interruption_overlap_ms"],
    )
    out_path = out_dir / "speaker_features.json"
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)
    print(f"[features] done -> {out_path}")
    for spk, vals in features.items():
        print(f"  {spk}: talk={vals['speaking_time_s']:.0f}s "
              f"turns={vals['turn_count']} "
              f"interruptions={vals['interruptions_made']}")


STAGES = {"audio": stage_audio, "features": stage_features}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True,
                    help="Session ID, e.g. s01_groupA (expects data/<id>.wav)")
    ap.add_argument("--stage", required=True, choices=STAGES)
    args = ap.parse_args()

    cfg = load_config()
    STAGES[args.stage](args.session, cfg)


if __name__ == "__main__":
    main()
