# ============================================================
# LEAK PREVENTION — human-subjects data NEVER enters git.
# If you think something here should be committed, ask the
# repo owner first. When in doubt, it stays out.
# ============================================================

# --- Participant data & derivatives (IRB-protected) ---
data/*
!data/.gitkeep
!data/README.md
output/*
!output/.gitkeep
*.wav
*.mp3
*.mp4
*.mov
*.mkv
*.flac
*.m4a
*transcript*.json
*diarization*

# --- Secrets & tokens ---
.env
*.token
*token*.txt
hf_*

# --- Large build artifacts ---
*.sif
*.img
models/
*.pt
*.pth
*.onnx
*.gguf

# --- Python ---
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
audio-env/
vision-env/
nlp-env/
.ipynb_checkpoints/

# --- Caches ---
.cache/
.huggingface/
wandb/
.dvc/cache

# --- OS / editor noise ---
.DS_Store
Thumbs.db
*.swp
