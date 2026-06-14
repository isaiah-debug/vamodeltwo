"""
Video → dialogue pipeline.

Full chain on one or more MP4 files:
  1. ffmpeg         — extract 16 kHz mono WAV per video
  2. WhisperX       — GPU transcription + word-level timestamps (RTX 5060)
  3. pyannote       — speaker diarization  (who is speaking when, audio)
  4. InsightFace    — face detection + embedding + track clustering (GPU)
  5. MediaPipe      — mouth-openness active-speaker detection (CPU, fast)
  6. Mapper         — align audio speaker IDs with visual face clusters
                      → named turns matching the existing pipeline format

GPU usage breakdown:
  WhisperX large-v3-turbo   ~3 GB VRAM  (float16, CTranslate2 backend)
  InsightFace buffalo_l      ~1 GB VRAM
  pyannote diarization       ~1 GB VRAM
  Total peak                 ~5-6 GB  (well within RTX 5060 8 GB)

Each stage is independently callable so you can resume after a crash or
skip stages that are already done (outputs are cached to disk).

Headcam shortcut
────────────────
If each video is a headcam for ONE known player, set camera_to_player:
    camera_to_player = {"session_A.mp4": "A", "session_B.mp4": "B", ...}
Face / diarization stages are skipped and speaker label = player label.

Usage
─────
    from src.video_pipeline import process_session
    turns = process_session(
        video_paths=["data/videos/cam_A.mp4", "data/videos/cam_B.mp4"],
        camera_to_player={"cam_A.mp4": "A", "cam_B.mp4": "B"},
        out_dir="output/session_01",
        whisper_model="large-v3-turbo",
    )
    # turns is a list[dict] compatible with annotate_turns / build_explicit_graph
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# ── Lazy imports (heavy, GPU-dependent) ───────────────────────────────────────

def _whisperx():
    try:
        import whisperx
        return whisperx
    except ImportError:
        raise ImportError(
            "whisperx not installed. Run: install_video_stack.bat"
        )

def _insightface():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        return FaceAnalysis
    except ImportError:
        raise ImportError("insightface not installed.")

def _mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        raise ImportError("mediapipe not installed.")

def _torch():
    import torch
    return torch


# ── Stage 1: Audio extraction ──────────────────────────────────────────────────

def extract_audio(
    video_path: str | Path,
    out_dir: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """
    ffmpeg: mp4 → 16 kHz mono WAV.
    Cached: re-runs only if WAV is older than the video.
    """
    video_path = Path(video_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path   = out_dir / (video_path.stem + ".wav")

    if wav_path.exists() and wav_path.stat().st_mtime > video_path.stat().st_mtime:
        print(f"  [audio] cached → {wav_path.name}")
        return wav_path

    cmd = [
        "ffmpeg", "-y",
        "-i",          str(video_path),
        "-vn",                            # no video
        "-ar",         str(sample_rate),  # 16 kHz
        "-ac",         "1",              # mono
        "-c:a",        "pcm_s16le",
        str(wav_path),
    ]
    print(f"  [audio] extracting {video_path.name} → {wav_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return wav_path


# ── Stage 2: Transcription + diarization ──────────────────────────────────────

def transcribe(
    wav_path: str | Path,
    out_dir: str | Path,
    model_name: str = "large-v3-turbo",
    language: str = "en",
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 4,
    hf_token: str = "",
    batch_size: int = 16,
    compute_type: str = "float16",
) -> list[dict]:
    """
    WhisperX: WAV → word-aligned segments with SPEAKER_XX labels.
    Cached to <out_dir>/<stem>_transcript.json.
    Uses your RTX 5060 via CUDA.
    """
    wav_path = Path(wav_path)
    out_dir  = Path(out_dir)
    cache    = out_dir / (wav_path.stem + "_transcript.json")

    if cache.exists():
        print(f"  [whisperx] cached → {cache.name}")
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    wx = _whisperx()
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [whisperx] loading {model_name} on {device.upper()}...")

    model = wx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    audio = wx.load_audio(str(wav_path))
    print(f"  [whisperx] transcribing {wav_path.name}...")
    result = model.transcribe(audio, batch_size=batch_size)

    # Word-level alignment
    print("  [whisperx] aligning words...")
    align_model, metadata = wx.load_align_model(
        language_code=language, device=device
    )
    result = wx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    # Speaker diarization
    if hf_token:
        print("  [whisperx] diarizing speakers...")
        diarize_model = wx.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = wx.assign_word_speakers(diarize_segments, result)
    else:
        print("  [whisperx] WARNING: no HF_TOKEN — skipping diarization. "
              "Set HF_TOKEN env var to enable speaker labels.")

    # Flatten to turn-like dicts
    segments = result.get("segments", [])
    turns = [
        {
            "start_s":   seg.get("start", 0),
            "end_s":     seg.get("end", 0),
            "utterance": seg.get("text", "").strip(),
            "speaker":   seg.get("speaker", "SPEAKER_00"),
        }
        for seg in segments
        if seg.get("text", "").strip()
    ]

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2, ensure_ascii=False)
    print(f"  [whisperx] {len(turns)} segments → {cache.name}")
    return turns


# ── Stage 3: Face detection + embedding + clustering ──────────────────────────

def detect_faces(
    video_path: str | Path,
    out_dir: str | Path,
    sample_every_n_frames: int = 15,
    det_size: tuple = (640, 640),
) -> dict:
    """
    InsightFace: sample frames → face embeddings → cluster to identities.
    Returns {face_id: [(timestamp_s, bbox), ...]} for each unique face.
    Cached to <out_dir>/<stem>_faces.json.
    """
    import cv2
    from sklearn.cluster import DBSCAN

    video_path = Path(video_path)
    out_dir    = Path(out_dir)
    cache      = out_dir / (video_path.stem + "_faces.json")

    if cache.exists():
        print(f"  [faces] cached → {cache.name}")
        with open(cache) as f:
            return json.load(f)

    FaceAnalysis = _insightface()
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider",
                                                     "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    all_embeddings = []
    all_meta       = []

    print(f"  [faces] scanning {video_path.name} every {sample_every_n_frames} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n_frames == 0:
            faces = app.get(frame)
            ts = frame_idx / fps
            for face in faces:
                emb = face.embedding
                if emb is not None:
                    all_embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))
                    all_meta.append({
                        "ts": round(ts, 2),
                        "bbox": [int(x) for x in face.bbox.tolist()],
                    })
        frame_idx += 1
    cap.release()

    if not all_embeddings:
        print("  [faces] no faces detected")
        result = {}
    else:
        # Cluster face embeddings → unique identities
        X = np.stack(all_embeddings)
        labels = DBSCAN(eps=0.45, min_samples=3, metric="cosine").fit_predict(X)
        face_tracks: dict[str, list] = {}
        for label, meta in zip(labels, all_meta):
            if label == -1:
                continue  # noise
            key = f"FACE_{label:02d}"
            face_tracks.setdefault(key, []).append(meta)
        result = face_tracks

    with open(cache, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [faces] found {len(result)} unique faces → {cache.name}")
    return result


# ── Stage 4: Active speaker detection (mouth openness) ────────────────────────

def detect_active_speakers(
    video_path: str | Path,
    out_dir: str | Path,
    face_tracks: dict,
    sample_every_n_frames: int = 5,
    mouth_open_ratio: float = 0.04,
) -> dict[str, list[float]]:
    """
    MediaPipe FaceMesh: detect mouth-open ratio per face per frame.
    Returns {face_id: [timestamp_s, ...]} — times when that face was speaking.
    Cached to <out_dir>/<stem>_active.json.

    Mouth open ratio threshold (default 0.04): empirically calibrated;
    increase if false positives, decrease if misses.
    """
    import cv2

    video_path = Path(video_path)
    out_dir    = Path(out_dir)
    cache      = out_dir / (video_path.stem + "_active.json")

    if cache.exists():
        print(f"  [active] cached → {cache.name}")
        with open(cache) as f:
            return json.load(f)

    mp = _mediapipe()
    mp_face_mesh = mp.solutions.face_mesh

    # Build a reverse index: for each timestamp, which face_id is nearest
    face_at_time: dict[float, str] = {}
    for fid, appearances in face_tracks.items():
        for ap in appearances:
            face_at_time[ap["ts"]] = fid

    speaking_times: dict[str, list[float]] = {fid: [] for fid in face_tracks}

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    UPPER_LIP = [13, 312, 311, 310, 415, 308]
    LOWER_LIP = [14, 317, 402, 318, 324, 78]

    print(f"  [active] mouth detection {video_path.name}...")
    with mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=4,
        refine_landmarks=True, min_detection_confidence=0.5
    ) as mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                ts = round(frame_idx / fps, 2)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mesh.process(rgb)
                h, w = frame.shape[:2]

                if results.multi_face_landmarks:
                    for face_lm in results.multi_face_landmarks:
                        lm = face_lm.landmark
                        upper_y = np.mean([lm[i].y * h for i in UPPER_LIP])
                        lower_y = np.mean([lm[i].y * h for i in LOWER_LIP])
                        face_h  = abs(lm[10].y - lm[152].y) * h or 1
                        openness = abs(lower_y - upper_y) / face_h

                        if openness > mouth_open_ratio:
                            # Find closest tracked face at this timestamp
                            closest_ts = min(
                                (t for t in face_at_time if abs(t - ts) < 1.0),
                                key=lambda t: abs(t - ts),
                                default=None,
                            )
                            if closest_ts is not None:
                                fid = face_at_time[closest_ts]
                                speaking_times[fid].append(ts)
            frame_idx += 1
    cap.release()

    with open(cache, "w") as f:
        json.dump(speaking_times, f, indent=2)
    total = sum(len(v) for v in speaking_times.values())
    print(f"  [active] {total} active-speech frames detected → {cache.name}")
    return speaking_times


# ── Stage 5: Map audio speakers → visual face IDs → player labels ─────────────

def map_speakers_to_players(
    audio_turns: list[dict],
    face_tracks: dict,
    speaking_times: dict[str, list[float]],
    player_labels: Optional[list[str]] = None,
) -> dict[str, str]:
    """
    Align pyannote SPEAKER_XX labels with visual FACE_XX identities,
    then optionally map to human-readable player labels.

    Algorithm:
      For each audio speaker segment, count how many active-speech frames
      of each face_id overlap with that time window.
      The face_id with most overlap is the visual match.

    Returns {audio_speaker_id: player_label}.
    """
    audio_speakers = sorted(set(t["speaker"] for t in audio_turns))
    face_ids       = list(face_tracks.keys())

    # Build overlap matrix
    overlap: dict[str, dict[str, int]] = {
        sp: {fid: 0 for fid in face_ids} for sp in audio_speakers
    }

    # Index speaking times for fast lookup
    face_speaking_set: dict[str, set] = {
        fid: set(round(t, 1) for t in times)
        for fid, times in speaking_times.items()
    }

    for turn in audio_turns:
        sp  = turn["speaker"]
        t0  = turn["start_s"]
        t1  = turn["end_s"]
        for fid in face_ids:
            count = sum(
                1 for t in face_speaking_set[fid]
                if t0 <= t <= t1
            )
            overlap[sp][fid] += count

    # Hungarian-style greedy assignment
    mapping: dict[str, str] = {}
    assigned_faces: set[str] = set()

    for sp in sorted(audio_speakers, key=lambda s: -max(overlap[s].values(), default=0)):
        best_fid = max(
            (fid for fid in face_ids if fid not in assigned_faces),
            key=lambda fid: overlap[sp][fid],
            default=None,
        )
        if best_fid:
            mapping[sp] = best_fid
            assigned_faces.add(best_fid)

    # Apply human-readable labels
    if player_labels:
        face_to_player = {}
        sorted_faces = sorted(face_ids, key=lambda f: int(f.split("_")[-1]))
        for fid, label in zip(sorted_faces, player_labels):
            face_to_player[fid] = label
        return {sp: face_to_player.get(fid, fid) for sp, fid in mapping.items()}

    return mapping


# ── Stage 6: Assemble turns ────────────────────────────────────────────────────

def assemble_turns(
    audio_turns: list[dict],
    speaker_map: dict[str, str],
    video_source: str = "",
    session_id: str = "session",
) -> list[dict]:
    """
    Apply speaker labels and format into the pipeline's standard turn dict.
    """
    turns = []
    for i, seg in enumerate(audio_turns):
        raw_spk = seg.get("speaker", "UNKNOWN")
        player  = speaker_map.get(raw_spk, raw_spk)
        start_s = seg.get("start_s", 0) or 0
        turns.append({
            "turn":         i + 1,
            "speaker":      player,
            "utterance":    seg.get("utterance", ""),
            "timestamp":    f"{int(start_s // 60):02d}:{int(start_s % 60):02d}",
            "start_s":      start_s,
            "end_s":        seg.get("end_s", start_s),
            "receiver_raw": "",   # not available from audio alone
            "notes":        f"auto:{video_source}",
            "session":      session_id,
        })
    return turns


# ── Main entry point ───────────────────────────────────────────────────────────

def process_session(
    video_paths: list[str | Path],
    out_dir: str | Path = "output/video_session",
    session_id: str = "session_01",
    camera_to_player: Optional[dict[str, str]] = None,
    player_labels: Optional[list[str]] = None,
    whisper_model: str = "large-v3-turbo",
    language: str = "en",
    num_speakers: Optional[int] = None,
    min_speakers: int = 2,
    max_speakers: int = 4,
    hf_token: str = "",
    run_face_id: bool = True,
    sample_video_every_n: int = 15,
    progress_callback=None,
) -> list[dict]:
    """
    Full pipeline: list of video files → list of turn dicts.

    Parameters
    ----------
    video_paths       : One or more MP4 files. If each is a headcam for one
                        player, set camera_to_player to skip face ID.
    camera_to_player  : {"filename.mp4": "A", ...}  — headcam shortcut.
    player_labels     : ["A", "B", "C", "D"]  — ordered labels for face clusters.
    whisper_model     : WhisperX model name. Recommended: "large-v3-turbo".
    run_face_id       : False = use audio diarization only (faster).
    hf_token          : HuggingFace token for pyannote diarization.
                        Get free token at huggingface.co and accept
                        pyannote/speaker-diarization-3.1 terms.
    progress_callback : Optional fn(step: str, pct: float) for UI progress bars.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = hf_token or os.environ.get("HF_TOKEN", "")

    def _progress(step, pct=0.0):
        if progress_callback:
            progress_callback(step, pct)
        else:
            print(f"  [{pct:3.0%}] {step}")

    all_turns: list[dict] = []

    for vid_path in video_paths:
        vid_path = Path(vid_path)
        fname    = vid_path.name
        vid_out  = out_dir / vid_path.stem
        vid_out.mkdir(exist_ok=True)

        print(f"\n{'─'*50}")
        print(f"Processing: {fname}")
        print(f"{'─'*50}")

        # ── Headcam shortcut ─────────────────────────────────────────────────
        if camera_to_player and fname in camera_to_player:
            player = camera_to_player[fname]
            _progress(f"Extracting audio [{fname}]", 0.1)
            wav = extract_audio(vid_path, vid_out)

            _progress(f"Transcribing [{fname}] → Player {player}", 0.3)
            audio_turns = transcribe(
                wav, vid_out, model_name=whisper_model,
                language=language,
                num_speakers=1, min_speakers=1, max_speakers=1,
                hf_token="",       # single speaker — no diarization needed
                batch_size=16,
            )
            # All speech = player
            speaker_map = {seg["speaker"]: player for seg in audio_turns}
            turns = assemble_turns(audio_turns, speaker_map,
                                   video_source=fname, session_id=session_id)
            _progress(f"Done [{fname}]", 1.0)
            all_turns.extend(turns)
            continue

        # ── Full pipeline (security cameras / mixed speakers) ────────────────
        _progress(f"Extracting audio [{fname}]", 0.05)
        wav = extract_audio(vid_path, vid_out)

        _progress(f"Transcribing [{fname}]", 0.15)
        audio_turns = transcribe(
            wav, vid_out, model_name=whisper_model, language=language,
            num_speakers=num_speakers, min_speakers=min_speakers,
            max_speakers=max_speakers, hf_token=hf_token, batch_size=16,
        )

        if run_face_id:
            _progress(f"Detecting faces [{fname}]", 0.45)
            face_tracks = detect_faces(vid_path, vid_out,
                                       sample_every_n_frames=sample_video_every_n)

            _progress(f"Active-speaker detection [{fname}]", 0.65)
            speaking_times = detect_active_speakers(vid_path, vid_out, face_tracks)

            _progress(f"Mapping speakers → faces [{fname}]", 0.80)
            speaker_map = map_speakers_to_players(
                audio_turns, face_tracks, speaking_times, player_labels
            )
        else:
            # Audio-only: map SPEAKER_00 → A, SPEAKER_01 → B, etc.
            raw_speakers = sorted(set(t["speaker"] for t in audio_turns))
            labels = player_labels or [f"SPK_{i}" for i in range(len(raw_speakers))]
            speaker_map = dict(zip(raw_speakers, labels))

        _progress(f"Assembling turns [{fname}]", 0.90)
        turns = assemble_turns(audio_turns, speaker_map,
                               video_source=fname, session_id=session_id)
        _progress(f"Done [{fname}]", 1.0)
        all_turns.extend(turns)

    # Sort all turns by timestamp across all videos
    all_turns.sort(key=lambda t: t.get("start_s") or 0)
    for i, t in enumerate(all_turns):
        t["turn"] = i + 1

    out_file = out_dir / "video_turns.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_turns, f, indent=2, ensure_ascii=False)
    print(f"\n[video_pipeline] {len(all_turns)} total turns → {out_file}")
    return all_turns
