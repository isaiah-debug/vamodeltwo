# Laguna workflow for multi-track MP4 transcription

Use Laguna only when you are ready to run transcription. Notebook setup and job
review do not consume GPUs.

## Expected input

- 7 MP4 files
- 4 isolated audio tracks per MP4
- one audio track per participant
- reference face images for A/B/C/D under the paths in the session config

Speaker identity comes from the track map. The cluster job runs:

```bash
python3 scripts/transcribe_to_csv.py --config configs/session_01.yaml
```

## Existing media location

The pilot MP4s are already on Laguna at:

```text
/project/SZhou_1896/Pilot Test Apr 2026
```

Do not copy them into the git repository. Point the session config at that
directory:

```yaml
videos:
  data_dir: "/project/SZhou_1896/Pilot Test Apr 2026"
```

Because the path contains spaces, quote it in shell commands:

```bash
ls "/project/SZhou_1896/Pilot Test Apr 2026"/*.mp4
```

Use the filenames printed by `ls` under `videos.files` in
`configs/session_01.yaml`.

## Jupyter workflow

Open:

```text
notebooks/laguna_transcript_workflow.ipynb
```

The notebook defaults to:

```python
SUBMIT = False
```

First run generates and prints a Slurm script. It does not call `sbatch`.

After reviewing paths, resources, and the config, set:

```python
SUBMIT = True
```

and rerun the submit cell.

## Terminal dry-run

```bash
python3 scripts/laguna_submit.py \
  --session session_01 \
  --time 08:00:00 \
  --cpus 8 \
  --mem 48G \
  --gpus 1 \
  -- --config configs/session_01.yaml
```

## Terminal submit

```bash
python3 scripts/laguna_submit.py --submit \
  --session session_01 \
  --time 08:00:00 \
  --cpus 8 \
  --mem 48G \
  --gpus 1 \
  -- --config configs/session_01.yaml
```

## Static Slurm template

If you prefer a plain Slurm file:

```bash
sbatch --export=SESSION=session_01,TRANSCRIBE_ARGS="--config configs/session_01.yaml" scripts/hpc_submit.slurm
```

## Monitor and cancel

```bash
squeue -u "$USER"
scancel <job_id>
```

Logs are written under:

```text
output/slurm/
```

The final CSV is:

```text
output/<session_id>/utterances.csv
```
