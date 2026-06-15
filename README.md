# Multi-track MP4 Dialogue Transcriber

This project has one purpose:

> Input **7 MP4 files**, each containing **4 isolated audio tracks** (one track
> per participant), and output a CSV of speaker dialogue with timestamps,
> speaker identity, inferred addressee, and transcript text.

The main artifact is:

```text
output/<session_id>/utterances.csv
```

## Output schema

```csv
session,turn,source_file,audio_track,start_s,end_s,local_start_s,local_end_s,timestamp,speaker,addressee,utterance,addressee_method
```

Important columns:

- `source_file`: MP4 chunk where the utterance came from.
- `audio_track`: MP4 audio stream index, e.g. `0` for `0:a:0`.
- `start_s` / `end_s`: session-level timestamps, including any file offset.
- `local_start_s` / `local_end_s`: timestamp inside the source MP4.
- `speaker`: participant mapped from the audio track.
- `addressee`: who the utterance appears directed to.
- `addressee_method`: provenance for that addressee label.

## Why no diarization or face recognition?

The input already has one isolated audio track per speaker. Speaker identity is
therefore a deterministic track mapping:

```text
0:a:0 -> A
0:a:1 -> B
0:a:2 -> C
0:a:3 -> D
```

The tool does not run diarization, face detection, graph analysis, leadership
scoring, dashboards, or visualization. Those were removed because they do not
help produce the requested CSV.

## Setup

Core utilities:

```bash
python3 -m pip install -r requirements.txt
```

Media transcription:

```bash
python3 -m pip install -r requirements-video.txt
```

Also install `ffmpeg` through your OS or cluster environment.

## Configure a session

Copy and edit:

```bash
cp configs/project_template.yaml configs/session_01.yaml
```

The key parts are:

```yaml
videos:
  data_dir: "data/videos"
  expected_files: 7
  audio_tracks:
    - index: 0
      player: "A"
    - index: 1
      player: "B"
    - index: 2
      player: "C"
    - index: 3
      player: "D"
  files:
    - file: "session_part1.mp4"
      offset_s: 0
    - file: "session_part2.mp4"
      offset_s: 1800
```

Use `offset_s` when the seven MP4s are sequential chunks and you want one
global session clock in the CSV.

## Run locally

From config:

```bash
python3 scripts/transcribe_to_csv.py --config configs/session_01.yaml
```

Or directly:

```bash
python3 scripts/transcribe_to_csv.py \
  --media data/videos/session_part1.mp4 data/videos/session_part2.mp4 data/videos/session_part3.mp4 data/videos/session_part4.mp4 data/videos/session_part5.mp4 data/videos/session_part6.mp4 data/videos/session_part7.mp4 \
  --track-map 0=A 1=B 2=C 3=D \
  --player-name A=Jordan B=Elis C=Anna D=Isaiah \
  --file-offset session_part1.mp4=0 session_part2.mp4=1800 session_part3.mp4=3600 session_part4.mp4=5400 session_part5.mp4=7200 session_part6.mp4=9000 session_part7.mp4=10800 \
  --session session_01 \
  --out output/session_01/utterances.csv
```

## Run on Laguna from Jupyter

Use:

```text
notebooks/laguna_transcript_workflow.ipynb
```

The notebook is safe by default:

1. It calls `scripts/laguna_submit.py` to write and print a Slurm script.
2. It does not submit while `SUBMIT = False`.
3. It submits only after you explicitly set `SUBMIT = True` or pass `--submit`.

Dry-run from a terminal:

```bash
python3 scripts/laguna_submit.py \
  --session session_01 \
  --time 08:00:00 \
  --cpus 8 \
  --mem 48G \
  --gpus 1 \
  -- --config configs/session_01.yaml
```

Submit only after reviewing the generated Slurm script:

```bash
python3 scripts/laguna_submit.py --submit \
  --session session_01 \
  --time 08:00:00 \
  --cpus 8 \
  --mem 48G \
  --gpus 1 \
  -- --config configs/session_01.yaml
```

## Addressee inference

Audio tracks identify who is speaking, not who they are speaking to. The tool
adds an auditable `addressee_method`:

| Method | Meaning |
| --- | --- |
| `broadcast_keyword` | Words like "everyone", "you guys", "team" |
| `name_mention` | Participant name from `--player-name` or config |
| `pronoun_context` | "you/your" resolved to nearest distinct speaker |
| `sequential_context` | Fallback to next/previous distinct speaker |
| `unknown` | No addressee could be inferred |

Use `--no-sequential-addressee` if you want weak fallbacks left blank.

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Data policy

Do not commit participant media, transcripts, outputs, model weights, or tokens.
The root `.gitignore` excludes those artifacts.
