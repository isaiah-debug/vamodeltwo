#!/bin/bash
# ============================================================
# HPC batch script — Laguna cluster (SLURM)
# Runs the full psych-graph pipeline on a session from Excel.
#
# Usage:
#   sbatch scripts/hpc_submit.slurm --export=EXCEL=data/session_01.xlsx,SESSION=s01
#   (or edit EXCEL / SESSION below and submit directly)
#
# Transfer data to Laguna:
#   rsync -avz data/ <user>@laguna.hpc.ucsd.edu:~/behavior-pipeline/data/
#   rsync -avz src/ app.py requirements.txt <user>@laguna.hpc.ucsd.edu:~/behavior-pipeline/
# ============================================================

#SBATCH --job-name=psych-graph
#SBATCH --output=output/slurm_%j_%x.out
#SBATCH --error=output/slurm_%j_%x.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ── Config ────────────────────────────────────────────────────────────────────
EXCEL="${EXCEL:-data/sample_dialogue.xlsx}"
SESSION="${SESSION:-session_01}"
REPO="$HOME/behavior-pipeline"
VENV="$REPO/psych-graph-env"

# ── Environment ───────────────────────────────────────────────────────────────
source "$VENV/bin/activate"
export PYTHONPATH="$REPO/src:$PYTHONPATH"

echo "============================================================"
echo " Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo " Excel: $EXCEL   Session: $SESSION"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

python - <<'PYEOF'
import os, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["REPO"]).parent / "src"))

from excel_parser import load_dialogue, save_json
from social_graph import build_graph, compute_metrics, erdos_distances, export_html, export_metrics_csv
from psych_analysis import annotate_turns, speaker_profiles, emotional_contagion_matrix

excel  = os.environ["EXCEL"]
sid    = os.environ["SESSION"]
outdir = Path("output") / sid
outdir.mkdir(parents=True, exist_ok=True)

print(f"[1/4] Parsing Excel: {excel}")
turns = load_dialogue(excel, session_id=sid)
save_json(turns, outdir / "turns_raw.json")

print(f"[2/4] NLP annotation (GPU)")
turns = annotate_turns(turns, use_emotion_model=True, batch_size=64)
save_json(turns, outdir / "turns_annotated.json")

print(f"[3/4] Building social graph")
G       = build_graph(turns)
metrics = compute_metrics(G)
profiles = speaker_profiles(turns)
contagion = emotional_contagion_matrix(turns)

export_html(G, outdir / "graph.html", metrics=metrics)
export_metrics_csv(metrics, outdir / "metrics.csv")

with open(outdir / "profiles.json", "w") as f:
    json.dump(profiles, f, indent=2)
with open(outdir / "contagion.json", "w") as f:
    json.dump(contagion, f, indent=2)

dist_df = erdos_distances(G)
dist_df.to_csv(outdir / "erdos_distances.csv")

print(f"[4/4] Done — all outputs in {outdir}/")
PYEOF
