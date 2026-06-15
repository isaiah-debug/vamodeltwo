# Multi-track MP4 Dialogue Transcriber

This project has one purpose:

> Input **7 MP4 files**, each containing **4 isolated audio tracks** (one track
> per participant), use the video and facial recognition evidence to improve
> "who is speaking to whom", and output a reviewable dialogue CSV.

The main artifact is:

```text
output/<session_id>/utterances.csv
```

## Output schema

```csv
session,turn,source_file,audio_track,start_s,end_s,local_start_s,local_end_s,timestamp,speaker,addressee,visual_addressee,visual_confidence,visual_method,visual_votes,visual_evidence,utterance,addressee_method
```

Important columns:

- `source_file`: MP4 chunk where the utterance came from.
- `audio_track`: MP4 audio stream index, e.g. `0` for `0:a:0`.
- `start_s` / `end_s`: session-level timestamps, including any file offset.
- `local_start_s` / `local_end_s`: timestamp inside the source MP4.
- `speaker`: participant mapped from the audio track.
- `addressee`: who the utterance appears directed to.
- `visual_addressee`: face/gaze-derived addressee before text fallbacks.
- `visual_confidence`, `visual_method`, `visual_votes`, `visual_evidence`:
  audit fields for visual/facial recognition.
- `addressee_method`: provenance for that addressee label.

## What the tool uses

- **ffmpeg** extracts each participant's isolated audio stream.
- **WhisperX** transcribes each speaker track with timestamps.
- **InsightFace/OpenCV** sample video frames during each utterance, identify
  participant faces from reference images, and use face pose/relative position
  as evidence for who the speaker is addressing.
- **Rule-based language/context fallbacks** fill gaps when visual evidence is
  unavailable.

The input already has one isolated audio track per speaker, so speaker identity
starts from a deterministic track mapping:

```text
0:a:0 -> A
0:a:1 -> B
0:a:2 -> C
0:a:3 -> D
```

The tool does **not** run social graph analysis, leadership scoring, dashboards,
or network visualization. Those are out of scope for automating manual
transcription.

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

## Step-by-step: run one session

### 1. Create ignored data folders

For laptop tests, participant media and face images can live under `data/`,
which is ignored by git. On Laguna, the MP4s can stay in the existing
`/project/SZhou_1896/Pilot Test Apr 2026` directory.

```bash
mkdir -p data/videos/session_01
mkdir -p data/faces
mkdir -p output/session_01
```

### 2. Point to the seven MP4 files

If the MP4s are already on Laguna at:

```text
/project/SZhou_1896/Pilot Test Apr 2026
```

do not copy them into the repository. Use that absolute directory in the
session config:

```yaml
videos:
  data_dir: "/project/SZhou_1896/Pilot Test Apr 2026"
```

Because the path contains spaces, always quote it in shell commands:

```bash
ls "/project/SZhou_1896/Pilot Test Apr 2026"
```

If you are running on your laptop instead, copy from your laptop or external
drive:

```bash
cp "/path/to/session_part1.mp4" data/videos/session_01/
cp "/path/to/session_part2.mp4" data/videos/session_01/
cp "/path/to/session_part3.mp4" data/videos/session_01/
cp "/path/to/session_part4.mp4" data/videos/session_01/
cp "/path/to/session_part5.mp4" data/videos/session_01/
cp "/path/to/session_part6.mp4" data/videos/session_01/
cp "/path/to/session_part7.mp4" data/videos/session_01/
```

Or symlink from an external drive to avoid duplicating large files:

```bash
ln -s "/path/to/external/session_part1.mp4" data/videos/session_01/
```

### 3. Add face reference images

Use one clear image per participant:

```bash
cp "/path/to/player_A_face.jpg" data/faces/A.jpg
cp "/path/to/player_B_face.jpg" data/faces/B.jpg
cp "/path/to/player_C_face.jpg" data/faces/C.jpg
cp "/path/to/player_D_face.jpg" data/faces/D.jpg
```

### 4. Verify the audio tracks

On Laguna, first list the actual filenames:

```bash
ls "/project/SZhou_1896/Pilot Test Apr 2026"/*.mp4
```

Then run `ffprobe` on one MP4 before transcription:

```bash
ffprobe -v error -select_streams a \
  -show_entries stream=index:stream_tags=title \
  -of table "/project/SZhou_1896/Pilot Test Apr 2026/session_part1.mp4"
```

Confirm the audio streams match the config, for example:

```text
0:a:0 -> A
0:a:1 -> B
0:a:2 -> C
0:a:3 -> D
```

If the order is different, update `videos.audio_tracks` in the config.

### 5. Create and edit the session config

```bash
cp configs/project_template.yaml configs/session_01.yaml
```

Edit `configs/session_01.yaml`:

- Set `project.session_id`.
- Set each participant's `label` and `name`.
- Set `videos.data_dir` to the folder containing the MP4s. On Laguna, use
  `"/project/SZhou_1896/Pilot Test Apr 2026"`.
- Set the seven MP4 filenames under `videos.files`.
- Set `offset_s` for each file if the MP4s are sequential chunks.
- Set `visual.face_references` to `data/faces/A.jpg`, etc.

### 6. Fast smoke test without visual recognition

This confirms that config loading, MP4 audio-track extraction, and transcription
work before spending time on face recognition:

```bash
python3 scripts/transcribe_to_csv.py \
  --config configs/session_01.yaml \
  --no-visual
```

For the very fastest first test, temporarily set:

```yaml
videos:
  expected_files: 1
  files:
    - file: "session_part1.mp4"
      offset_s: 0
```

Restore all seven files after the smoke test.

### 7. Full run with visual/facial addressee evidence

```bash
python3 scripts/transcribe_to_csv.py --config configs/session_01.yaml
```

The output is:

```text
output/session_01/utterances.csv
```

### 8. Inspect the CSV

Open `output/session_01/utterances.csv` and check:

- `speaker` is A/B/C/D as expected.
- `utterance` text is readable.
- `addressee` is populated where possible.
- `addressee_method` shows whether the addressee came from `face_gaze`,
  `broadcast_keyword`, `name_mention`, `pronoun_context`, or another fallback.
- `visual_*` columns provide review evidence for face/gaze assignments.

## Configure a session

Copy and edit:

```bash
cp configs/project_template.yaml configs/session_01.yaml
```

The key parts are:

```yaml
videos:
  data_dir: "/project/SZhou_1896/Pilot Test Apr 2026"
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

Add reference face images so the visual layer can label A/B/C/D:

```yaml
visual:
  enabled: true
  sample_fps: 1.0
  identity_threshold: 0.35
  yaw_threshold: 12.0
  face_references:
    A: "data/faces/A.jpg"
    B: "data/faces/B.jpg"
    C: "data/faces/C.jpg"
    D: "data/faces/D.jpg"
```

Use clear, front-facing reference images. Store them outside git under
`data/faces/`.

For debugging transcription without the visual pass, run with `--no-visual`.

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
  --visual \
  --face-reference A=data/faces/A.jpg B=data/faces/B.jpg C=data/faces/C.jpg D=data/faces/D.jpg \
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

Step by step:

1. Copy or sync this repository to Laguna.
2. Confirm the MP4s are visible at
   `/project/SZhou_1896/Pilot Test Apr 2026`.
3. Copy/sync `data/faces/` to Laguna if the face reference images are not
   already there.
4. Make sure `configs/session_01.yaml` uses:

   ```yaml
   videos:
     data_dir: "/project/SZhou_1896/Pilot Test Apr 2026"
   ```

5. Open `notebooks/laguna_transcript_workflow.ipynb`.
6. Set:

   ```python
   SESSION = "session_01"
   CONFIG = "configs/session_01.yaml"
   SUBMIT = False
   ```

7. Run the dry-run cell and review the generated Slurm script.
8. When the script looks correct, set `SUBMIT = True`.
9. Run the submit cell.
10. Monitor:

   ```bash
   squeue -u "$USER"
   ```

11. After the job finishes, inspect:

    ```text
    output/session_01/utterances.csv
    ```

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

Audio tracks identify who is speaking. Video/facial recognition and text/context
features estimate who they are speaking to. The tool adds an auditable
`addressee_method`:

| Method | Meaning |
| --- | --- |
| `face_gaze` | Visual face recognition and speaker head-pose evidence |
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
