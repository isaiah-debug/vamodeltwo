# behavior-pipeline

Converts multi-camera / multi-microphone group recordings into psychological
measures of leadership emergence: speaking time, turn-taking, interruptions,
prosody, content codes, and who-responds-to-whom networks.

**New to the project? Read [`docs/behavior_pipeline_guide.pdf`](docs/behavior_pipeline_guide.pdf) first** —
it explains every tool and how the stages fit together, assuming no background.

## Quickstart (one session, on your machine)

```bash
# 1. One-time setup (Ubuntu / WSL2) — installs all three environments
./environment/setup_behavior_pipeline.sh

# 2. Set your HuggingFace token (needed for speaker diarization)
cp .env.example .env        # then edit .env with your token

# 3. Put a session's audio in data/ (see data/README.md for naming)
#    Extract audio from video if needed:
ffmpeg -i data/s01_groupA_cam1.mp4 -vn -ar 16000 -ac 1 data/s01_groupA.wav

# 4. Run the audio stage
source ~/behavior-pipeline/scripts/activate-audio.sh
python scripts/process_one_session.py --session s01_groupA --stage audio

# 5. Inspect output/s01_groupA/ — speaker-labeled, timestamped transcript JSON
```

## Repo map

| Path | What lives here |
|---|---|
| `src/pipeline/` | Library code (importable, tested). Notebooks call this; logic lives here. |
| `scripts/` | Entry points you actually run. `process_one_session.py` is the unit everything is built around. |
| `scripts/slurm/` | Laguna cluster batch scripts (job arrays). |
| `configs/` | Everything you might change without reading code: paths, model sizes, the LLM codebook. |
| `containers/` | Apptainer `.def` recipes. Built `.sif` files are NOT committed (too large). |
| `environment/` | Laptop setup script + pinned requirements per environment. |
| `notebooks/` | Numbered exploration notebooks. Keep logic in `src/`, not here. |
| `docs/` | The guide PDF, laptop setup (SETUP.md), cluster deployment (LAGUNA.md). |
| `data/`, `output/` | Empty in git, **always**. See Data policy. |

## Data policy (read before your first commit)

**No participant data ever enters this repository.** The `.gitignore`
blocks `data/`, `output/`, all audio/video formats, transcripts, model
weights, and tokens. This is IRB-protected human-subjects data; treat a
push containing recordings as a reportable incident, not an oops.

Real data lives in lab storage — see `data/README.md` for location and
access. Tokens live in `.env` (gitignored); never hardcode them.

## Laptop vs. cluster

Develop and validate locally on 1–2 sessions; run the full dataset on
Laguna via the job array. Same containers, same code, different scale.
See `docs/LAGUNA.md`.

## Contributing (lab members)

1. Branch per feature: `git checkout -b feat/interruption-detection`
2. Logic goes in `src/pipeline/`, with a test in `tests/`
3. Run `pytest` before pushing
4. PR with a one-paragraph description of what changed and why
