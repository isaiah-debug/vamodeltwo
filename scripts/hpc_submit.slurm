#!/bin/bash
# Laguna Slurm template for the participant transcript CSV tool.
#
# Prefer scripts/laguna_submit.py from a notebook because it writes a reviewed
# job script first. This static template is for users who want to edit and
# submit a single file manually:
#
#   sbatch scripts/hpc_submit.slurm
#
# No compute is used until sbatch is called.

#SBATCH --job-name=transcript-csv
#SBATCH --output=output/slurm/%x_%j.out
#SBATCH --error=output/slurm/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$HOME/behavior-pipeline}"
SESSION="${SESSION:-session_01}"
TRANSCRIBE_ARGS="${TRANSCRIBE_ARGS:---config configs/project_template.yaml}"
VENV="${VENV:-}"
CONDA_ENV="${CONDA_ENV:-}"

cd "$REPO"
mkdir -p output/slurm "output/$SESSION"

echo "Laguna transcript CSV job ${SLURM_JOB_ID:-manual}"
echo "Repository : $REPO"
echo "Session    : $SESSION"
echo "Args       : $TRANSCRIBE_ARGS"

if [[ -n "$VENV" ]]; then
  source "$VENV/bin/activate"
elif [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$REPO:$REPO/src:${PYTHONPATH:-}"

# shellcheck disable=SC2086
python3 scripts/transcribe_to_csv.py $TRANSCRIBE_ARGS
