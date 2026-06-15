# Participant Transcript CSV Tool

This repository is now focused on one task for behavioral experiments:

> Take video/audio from a session, identify each participant's utterances, infer
> who each utterance was directed to, and write one auditable CSV.

The core output is `utterances.csv`.

```csv
session,turn,start_s,end_s,timestamp,speaker,addressee,utterance,addressee_method
session_01,1,1,2,00:01,A,All,Which one do you want to do first?,coded
session_01,2,7,10,00:07,A,B,You want to split up?,coded
```

`addressee_method` is included because "speaking to whom" is partly inferred
when no human-coded receiver is available. Reviewers can filter weak heuristics
instead of treating them as ground truth.

## What the tool does

1. Accepts one of:
   - video/audio files supported by `ffmpeg`
   - existing `video_turns.json`
   - existing per-player coded CSV files
2. Transcribes media with WhisperX through `src/video_pipeline.py`
3. Labels speakers by one of these routes:
   - headcam mapping (`cam_A.mp4=A`) when each recording belongs to one player
   - diarization/face mapping for mixed-camera footage
   - existing `speaker` columns for coded CSVs
4. Infers or preserves addressees with `src/addressee_inference.py`
5. Writes the narrow CSV with `src/export_utterances.py`

The older leadership, graph, and dashboard modules can still be inspected, but
they are not needed for this focused workflow.

## Quick start

Install core dependencies:

```bash
python -m pip install -r requirements.txt
```

For video/audio transcription, also install the video stack described in
`install_video_stack.bat` and make sure `ffmpeg` is on your `PATH`.

Speaker diarization for mixed-speaker audio requires a HuggingFace token with
access to pyannote models:

```bash
export HF_TOKEN=hf_yourtoken
```

## Run from headcam media

Use this when each file contains one known participant.

```bash
python scripts/transcribe_to_csv.py \
  --media data/videos/cam_A.mp4 data/videos/cam_B.mp4 data/videos/cam_C.mp4 data/videos/cam_D.mp4 \
  --camera-map cam_A.mp4=A cam_B.mp4=B cam_C.mp4=C cam_D.mp4=D \
  --player-name A=Jordan B=Elis C=Anna D=Isaiah \
  --session session_01 \
  --out output/session_01/utterances.csv
```

Headcam mapping skips diarization and face identification for those files.

## Run from mixed-camera media

Use this when a recording can contain multiple speakers.

```bash
python scripts/transcribe_to_csv.py \
  --media data/videos/sec_cam_1.mp4 \
  --players A B C D \
  --player-name A=Jordan B=Elis C=Anna D=Isaiah \
  --hf-token "$HF_TOKEN" \
  --session session_01 \
  --out output/session_01/utterances.csv
```

Add `--no-face` to use audio diarization only.

## Run from an existing project config

```bash
python scripts/transcribe_to_csv.py --config configs/project_template.yaml
```

The default output is:

```text
output/<session_id>/utterances.csv
```

## Run from existing coded CSVs

This is useful for validation and for sessions that have already been coded by
researchers.

```bash
python scripts/transcribe_to_csv.py \
  --from-csv data \
  --session escape_room_01 \
  --out output/escape_room_01/utterances.csv
```

Expected input columns are:

```csv
speaker,receiver,start,end,transcript,notes_challenges
```

Blank speaker cells are forward-filled by `src/multi_csv_loader.py`.

## Run from existing turn JSON

```bash
python scripts/transcribe_to_csv.py \
  --from-json output/session_01/video_turns.json \
  --out output/session_01/utterances.csv
```

This path does not require WhisperX or GPU dependencies.

## Addressee inference rules

Human-coded receivers are preserved first. When `receiver_raw` is empty, the
tool uses conservative, auditable heuristics:

| Method | Meaning |
| --- | --- |
| `coded` | Human-coded `receiver` / `receiver_raw` was present |
| `broadcast_keyword` | Utterance contains words like "everyone" or "you guys" |
| `name_mention` | Utterance names a participant from `--player-name` |
| `pronoun_context` | "you/your" resolved to the nearest distinct speaker |
| `sequential_context` | Fallback to next/previous distinct speaker |
| `unknown` | No addressee could be inferred |

Use `--no-sequential-addressee` if you prefer unknown over the weakest fallback.

## Test

```bash
pytest
```

The tests exercise addressee inference, CSV export, timestamp parsing, and coded
CSV loading without downloading models or running GPU transcription.

## Data policy

Do not commit participant media, transcripts, tokens, model weights, or derived
outputs. The repository `.gitignore` excludes `data/`, `output/`, media files,
transcript JSON, tokens, and model artifacts.
