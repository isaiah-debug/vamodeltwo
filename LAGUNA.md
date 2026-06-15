# Laguna notebook workflow

Use Laguna for transcription jobs when you need cluster GPUs, but keep setup and
review work lightweight. The repository is structured so a Jupyter notebook can
prepare a Slurm job without consuming compute; the job starts only after an
explicit submit step.

## Principle

1. Edit the session config and notebook parameters.
2. Generate a Slurm script in dry-run mode.
3. Review the printed command and requested resources.
4. Submit only when ready.
5. Monitor with `squeue`; cancel with `scancel` if needed.

## One-time cluster setup

On Laguna, clone or sync the repository and install the optional video stack in
an environment that includes `ffmpeg`, PyTorch, WhisperX, and pyannote access.

```bash
cd ~/behavior-pipeline
python3 -m pip install -r requirements-video.txt
```

If you use a virtual environment:

```bash
python3 -m venv ~/behavior-pipeline/transcript-env
source ~/behavior-pipeline/transcript-env/bin/activate
python3 -m pip install -r requirements-video.txt
```

For mixed-speaker diarization, set your HuggingFace token in the job
environment, not in source code:

```bash
export HF_TOKEN=hf_yourtoken
```

## From Jupyter: prepare, then submit

The notebook should call `scripts/laguna_submit.py`. Its default behavior is a
dry run: it writes and prints the Slurm script but does not call `sbatch`.

```python
from pathlib import Path
import subprocess

REPO = Path.cwd()
if not (REPO / "scripts" / "laguna_submit.py").exists():
    REPO = REPO.parent

SESSION = "session_01"
cmd = [
    "python3", "scripts/laguna_submit.py",
    "--session", SESSION,
    "--workdir", str(REPO),
    "--time", "04:00:00",
    "--cpus", "8",
    "--mem", "32G",
    "--gpus", "1",
    "--venv", str(Path.home() / "behavior-pipeline" / "transcript-env"),
    "--",
    "--config", "configs/project_template.yaml",
]
subprocess.run(cmd, check=True)
```

After reviewing the generated Slurm script, submit by adding `--submit` before
the `--` delimiter:

```python
submit_cmd = cmd[:2] + ["--submit"] + cmd[2:]
subprocess.run(submit_cmd, check=True)
```

## Common resource profiles

| Input type | Suggested flags | Notes |
| --- | --- | --- |
| Existing JSON or coded CSV | `--gpus 0 --cpus 2 --mem 8G --time 00:30:00` | No transcription model needed |
| Headcam media | `--gpus 1 --cpus 8 --mem 32G --time 04:00:00` | Uses WhisperX; skips face ID per mapped camera |
| Mixed-camera media | `--gpus 1 --cpus 8 --mem 48G --time 06:00:00` | Uses diarization and optional face mapping |

These settings request one job, not the whole cluster. Increase only when the
actual session data requires it.

## Monitor jobs

```bash
squeue -u "$USER"
```

View logs after a job starts:

```bash
ls output/slurm
```

Cancel a job:

```bash
scancel <job_id>
```

## Static Slurm template

If you do not want to use the notebook helper, edit and submit:

```bash
sbatch scripts/hpc_submit.slurm
```

The static template runs `scripts/transcribe_to_csv.py` and accepts environment
overrides:

```bash
sbatch --export=SESSION=session_01,REPO=$HOME/behavior-pipeline,VENV=$HOME/behavior-pipeline/transcript-env,TRANSCRIBE_ARGS="--config configs/project_template.yaml" scripts/hpc_submit.slurm
```

## Output

The expected artifact is:

```text
output/<session_id>/utterances.csv
```

If you pass `--out` inside the transcribe arguments, use that path instead.
