"""Multi-track MP4 to speaker-attributed dialogue turns.

The experiment input is seven long MP4 files. Each MP4 contains four isolated
audio tracks, one per participant. That means speaker identity comes from the
audio stream mapping; diarization and face recognition are deliberately out of
scope for this focused tool.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping, Optional


_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[object, object, object, str]] = {}


def extract_audio_track(
    media_path: str | Path,
    out_dir: str | Path,
    track_index: int,
    sample_rate: int = 16000,
) -> Path:
    """Extract one MP4 audio stream as 16 kHz mono WAV."""
    media_path = Path(media_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{media_path.stem}_track{track_index}.wav"

    if wav_path.exists() and wav_path.stat().st_mtime > media_path.stat().st_mtime:
        print(f"  [ffmpeg] cached {wav_path.name}")
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-map",
        f"0:a:{track_index}",
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    print(f"  [ffmpeg] extracting {media_path.name} audio track {track_index}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while extracting audio track {track_index} from "
            f"{media_path}:\n{result.stderr}"
        )
    return wav_path


def transcribe_isolated_wav(
    wav_path: str | Path,
    out_dir: str | Path,
    speaker: str,
    source_file: str,
    track_index: int,
    session_id: str,
    offset_s: float = 0.0,
    model_name: str = "large-v3-turbo",
    language: str = "en",
    batch_size: int = 16,
    compute_type: str = "float16",
) -> list[dict]:
    """Transcribe one isolated speaker WAV into normalized turn dictionaries."""
    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    cache = out_dir / f"{wav_path.stem}_turns.json"

    if cache.exists():
        print(f"  [whisperx] cached {cache.name}")
        with cache.open(encoding="utf-8") as handle:
            return json.load(handle)

    wx, model, align_model, metadata, device = _load_whisperx_models(
        model_name=model_name,
        language=language,
        compute_type=compute_type,
    )

    audio = wx.load_audio(str(wav_path))
    print(f"  [whisperx] transcribing {wav_path.name} as speaker {speaker}")
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

        local_start = float(segment.get("start", 0.0) or 0.0)
        local_end = float(segment.get("end", local_start) or local_start)
        start_s = offset_s + local_start
        end_s = offset_s + local_end
        turns.append(
            {
                "session": session_id,
                "source_file": source_file,
                "audio_track": track_index,
                "speaker": speaker,
                "utterance": utterance,
                "timestamp": seconds_to_timestamp(start_s),
                "start_s": start_s,
                "end_s": end_s,
                "local_start_s": local_start,
                "local_end_s": local_end,
                "receiver_raw": "",
                "notes": f"auto:{source_file}:track{track_index}",
            }
        )

    with cache.open("w", encoding="utf-8") as handle:
        json.dump(turns, handle, indent=2, ensure_ascii=False)
    print(f"  [whisperx] {len(turns)} turns -> {cache.name}")
    return turns


def process_multitrack_session(
    media_paths: list[str | Path],
    track_map: Mapping[int, str],
    out_dir: str | Path = "output/session_01",
    session_id: str = "session_01",
    file_offsets: Optional[Mapping[str, float]] = None,
    whisper_model: str = "large-v3-turbo",
    language: str = "en",
    batch_size: int = 16,
    compute_type: str = "float16",
    sample_rate: int = 16000,
) -> list[dict]:
    """Process all MP4 files and mapped speaker tracks into one turn timeline."""
    if not track_map:
        raise ValueError("track_map is required, e.g. {0: 'A', 1: 'B', 2: 'C', 3: 'D'}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_offsets = file_offsets or {}
    all_turns: list[dict] = []

    for media_path in media_paths:
        media_path = Path(media_path)
        media_out = out_dir / media_path.stem
        media_out.mkdir(parents=True, exist_ok=True)
        offset = float(file_offsets.get(media_path.name, file_offsets.get(str(media_path), 0.0)))

        print(f"\nProcessing {media_path.name} (offset={offset:g}s)")
        for track_index, speaker in sorted(track_map.items()):
            wav_path = extract_audio_track(
                media_path=media_path,
                out_dir=media_out,
                track_index=int(track_index),
                sample_rate=sample_rate,
            )
            all_turns.extend(
                transcribe_isolated_wav(
                    wav_path=wav_path,
                    out_dir=media_out,
                    speaker=str(speaker),
                    source_file=media_path.name,
                    track_index=int(track_index),
                    session_id=session_id,
                    offset_s=offset,
                    model_name=whisper_model,
                    language=language,
                    batch_size=batch_size,
                    compute_type=compute_type,
                )
            )

    all_turns.sort(key=lambda turn: (float(turn.get("start_s", 0.0)), str(turn.get("speaker", ""))))
    for index, turn in enumerate(all_turns, start=1):
        turn["turn"] = index

    out_file = out_dir / "turns.json"
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(all_turns, handle, indent=2, ensure_ascii=False)
    print(f"\n[video_pipeline] {len(all_turns)} turns -> {out_file}")
    return all_turns


def seconds_to_timestamp(seconds: float) -> str:
    total = int(float(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    remainder = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


def _load_whisperx_models(
    model_name: str,
    language: str,
    compute_type: str,
) -> tuple[object, object, object, object, str]:
    key = (model_name, language, compute_type, _device())
    if key in _MODEL_CACHE:
        wx, model, align_model, metadata = _MODEL_CACHE[key]
        return wx, model, align_model, metadata, key[-1]

    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "whisperx is required for MP4 transcription. Install the optional "
            "video stack with: python3 -m pip install -r requirements-video.txt"
        ) from exc

    device = key[-1]
    print(f"  [whisperx] loading {model_name} on {device}")
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    _MODEL_CACHE[key] = (whisperx, model, align_model, metadata)
    return whisperx, model, align_model, metadata, device


def _device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
