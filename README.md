# vamodeltwo — Psychological Dialogue Analysis

Turns group experiment footage (or manually coded dialogue sheets) into a
leadership emergence analysis: social network graph, Full Range Leadership
scores, emotion timelines, and turn-by-turn interaction maps.

---

## What this does vs. "training a model"

**Short answer: this uses pre-trained models, not from-scratch training —
and that is intentional and better for this use case.**

| Approach | What it means | When you'd do it |
| --- | --- | --- |
| **This project** | Download expert models others already trained (Whisper, emotion classifier, face detector), run them on your data, apply psych frameworks on top | You want results now, your GPU is for inference not weeks of training, your dataset is small |
| **Train from scratch** | Feed labeled examples into PyTorch until it learns your task | You have 100k+ labeled examples, your task is unique enough that no existing model covers it |
| **Fine-tune** | Start from a pre-trained model, update it on your labeled escape-room data | ~50–200 labeled examples, you want better accuracy on your specific speakers/vocab |

PyTorch IS running here — it's the engine underneath WhisperX and the
emotion classifier. But we're doing inference, not training. Your RTX 5060
runs the models in real time; a training run on this dataset would take hours
to days and need thousands of labeled examples you don't have yet.

Fine-tuning is the realistic next step once you have enough labeled sessions:
feed your coded CSVs back in as training data for a leadership-moment
classifier. That's a future feature.

---

## Run locally — step by step

### First-time setup (do this once)

#### Step 1 — Install Python dependencies

Double-click `run.bat` to start the dashboard. If it fails saying packages
are missing, open a terminal in this folder and run:

```bat
psych-graph-env\Scripts\activate
pip install -r requirements.txt
```

#### Step 2 — Install video stack (only if using MP4 input)

Double-click `install_video_stack.bat`

This installs WhisperX (GPU transcription), pyannote (speaker diarization),
InsightFace (face detection), and MediaPipe. Takes 5–10 minutes.

#### Step 3 — Get a free HuggingFace token (only needed for video mode)

Speaker diarization ("who is talking when") requires a free account:

1. Go to [huggingface.co](https://huggingface.co) — sign up (free)
2. Go to your profile → Settings → Access Tokens → New token (read)
3. Copy the token (starts with `hf_...`)
4. Accept the two model license agreements:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
5. Paste the token into the HuggingFace token field in the dashboard sidebar,
   or set it once as an environment variable: `set HF_TOKEN=hf_yourtoken`

You do not need HuggingFace for Excel/CSV mode.

---

### Every time — launch the dashboard

Option A: Double-click `run.bat`

Option B: Terminal

```bat
psych-graph-env\Scripts\activate
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`

---

### Loading your data

#### From Excel / CSV (manual dialogue sheets)

1. Sidebar → Input mode → **Excel / CSV**
2. Upload your `.xlsx`, `.xls`, or `.csv` file
3. Required columns: `speaker`, `utterance`
4. Optional: `turn`, `timestamp`, `receiver`, `code`
5. Click **Analyse**

The template is at `data/escape_room_dialogue_TEMPLATE.xlsx`.

#### From video files (auto-transcribe on GPU)

1. Complete Steps 2 and 3 above (install video stack + HF token)
2. Sidebar → Input mode → **Video files (auto-transcribe)**
3. Upload your MP4 files
4. If you have headcam footage (one camera per person), fill in the
   headcam map — e.g. `cam_A.mp4=A` — this skips face detection and
   is much faster
5. Paste your HF token
6. Click **Transcribe + Analyse**

For 4 × 30-minute headcam videos on an RTX 5060: roughly 8–15 minutes total.

#### From a previous run (reload)

1. Sidebar → Input mode → **Load saved JSON**
2. Upload the `*_annotated.json` from `output/<session>/`

---

### Command-line (batch / headless)

```bat
psych-graph-env\Scripts\activate
python scripts/process_videos.py --config configs/project_template.yaml
```

Copy `configs/project_template.yaml` for each new experiment and edit the
player names and video file paths. Output goes to `output/<session_id>/`.

---

## What each external tool does

| Tool | What it is | Why we use it |
| --- | --- | --- |
| **WhisperX** | OpenAI's Whisper speech-to-text, GPU-accelerated | Transcribes audio with word-level timestamps |
| **pyannote.audio** | Speaker diarization model | Figures out *who* is speaking at each moment |
| **InsightFace** | Face detection + recognition | Links a face track to a player across camera angles |
| **MediaPipe** | Google's mouth-openness detection | Confirms active speaker from video when audio is ambiguous |
| **HuggingFace** | Model host for pyannote and the emotion classifier | Free download of pre-trained weights |
| **VADER** | Rule-based sentiment (no GPU needed) | Fast per-turn positivity/negativity score |
| **distilRoBERTa emotion** | 7-class emotion classifier (joy/anger/sadness/fear/disgust/surprise/neutral) | Runs on your RTX 5060, batch inference |
| **NetworkX** | Graph library | Computes eigenvector centrality, PageRank, betweenness |
| **Plotly** | Interactive charting | The 6-panel visualization |
| **Streamlit** | Python web dashboard framework | The browser UI — no JavaScript needed |

---

## Output files

All outputs land in `output/<session_id>/` (gitignored):

| File | What it contains |
| --- | --- |
| `enhanced_analysis.html` | The full 6-panel interactive dashboard (open in any browser) |
| `turns_annotated.json` | Every turn with sentiment, emotion, leadership markers |
| `metrics.csv` | Per-speaker network centrality scores |
| `leadership_scores.json` | Full Range Leadership dimension scores per speaker |
| `leadership_moments.json` | Every tagged leadership moment with turn and type |
| `speaker_profiles.json` | Emotion distribution, verbal dominance markers |
| `erdos_distances.csv` | Social distance matrix |

---

## Data policy

No participant data ever enters this repository. `.gitignore` blocks `data/`,
`output/`, all audio/video formats, transcripts, model weights, and tokens.
Players are referred to only as A, B, C, D.

---

## Epilogue — how to start training your own model

Everything above uses other people's pre-trained models. This section is for
when you want a model that learns your specific experiment: your vocabulary,
your task environment, your leadership patterns. Here is the practical path.

### The language and tools

- **Language**: Python — same as everything else here
- **Framework**: PyTorch + HuggingFace `transformers` — both already installed in `psych-graph-env`
- **Key libraries**: `transformers`, `datasets`, `accelerate` (all from HuggingFace)
- **GPU**: Your RTX 5060 handles fine-tuning of small models (~200M parameters) in minutes

You do not need to change language or set up a new environment. Everything
runs in the same venv you already have.

### The task you would actually train

The most useful thing to train is a **leadership moment classifier**: given a
single utterance, predict whether it is a Directive, Insight, Consensus-building,
Coordination, Breakthrough, or none of the above.

Right now the pipeline uses hand-written regex patterns to tag these moments.
A trained classifier would be more accurate, more generalizable across
different groups and tasks, and would catch things the regex misses.

### What "enough data" looks like

| Sessions | Labeled turns | What you can do |
| --- | --- | --- |
| 1–3 (now) | ~500–1500 | Use regex + pre-trained models. Not enough to train. |
| 5–10 | ~2500–5000 | Fine-tune a small classifier. Noisy but useful. |
| 15–20 | ~7500–10000 | Reliable fine-tune. This is the realistic target. |
| 50+ | ~25000+ | Train from scratch if you wanted, but fine-tune is still better. |

The coded CSVs you already have (speaker, utterance, code columns) are your
training data. Every session you run adds to it.

### Step-by-step: fine-tune a leadership moment classifier

#### 1. Prepare your labeled data

Your existing CSVs already have an optional `code` column where you wrote
the interaction type (Directive, Insight, etc.). Export all of them into
a single file:

```python
import pandas as pd
from pathlib import Path

rows = []
for f in Path("data").glob("player_*.csv"):
    df = pd.read_csv(f)
    df = df[df["code"].notna() & (df["code"].str.strip() != "")]
    rows.append(df[["utterance", "code"]])

labeled = pd.concat(rows).dropna()
labeled.to_csv("data/training_labels.csv", index=False)
print(labeled["code"].value_counts())
```

#### 2. Create `scripts/train_classifier.py`

```python
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import pandas as pd
import numpy as np
import torch

# --- Load data ---
df = pd.read_csv("data/training_labels.csv")
labels = sorted(df["code"].unique().tolist())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

df["label"] = df["code"].map(label2id)
dataset = Dataset.from_pandas(df[["utterance", "label"]])
split = dataset.train_test_split(test_size=0.15, seed=42)

# --- Tokenize ---
MODEL = "distilbert-base-uncased"      # 66M params, fits in 8GB VRAM easily
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(batch["utterance"], truncation=True, padding="max_length", max_length=128)

split = split.map(tokenize, batched=True)

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# --- Training args (tuned for RTX 5060 8GB) ---
args = TrainingArguments(
    output_dir="models/leadership_classifier",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,                          # uses half-precision on your NVIDIA GPU
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
)

trainer.train()
trainer.save_model("models/leadership_classifier")
tokenizer.save_pretrained("models/leadership_classifier")
print("Saved to models/leadership_classifier")
```

Run it:

```bat
psych-graph-env\Scripts\activate
python scripts/train_classifier.py
```

Training time on RTX 5060: roughly **3–8 minutes** for 5000 labeled turns,
5 epochs.

#### 3. Plug the trained model back into the pipeline

In `src/leadership_assessment.py`, add a function that loads your trained
model and runs it instead of the regex:

```python
def classify_with_trained_model(turns, model_path="models/leadership_classifier"):
    from transformers import pipeline
    classifier = pipeline(
        "text-classification",
        model=model_path,
        device=0,          # GPU
        batch_size=64,
    )
    texts = [t["utterance"] for t in turns]
    predictions = classifier(texts)
    for turn, pred in zip(turns, predictions):
        turn["code_predicted"] = pred["label"]
        turn["code_confidence"] = round(pred["score"], 3)
    return turns
```

Then call it in `app.py` or `scripts/process_videos.py` after `annotate_turns`.

### What model to use

| Model | Size | VRAM needed | Best for |
| --- | --- | --- | --- |
| `distilbert-base-uncased` | 66M params | ~1 GB | Fast, good baseline, start here |
| `roberta-base` | 125M params | ~2 GB | Better accuracy, still fast |
| `distilroberta-base` | 82M params | ~1.5 GB | Already used for emotion; consistent choice |
| `bert-base-uncased` | 110M params | ~2 GB | Classic, well-documented |

Start with `distilbert-base-uncased`. Upgrade to `roberta-base` only if
accuracy on the test split is unsatisfying after 5+ epochs.

Do not use a model larger than `roberta-large` (355M) — it will not fit
in 8GB alongside the rest of the pipeline.

### What you are NOT doing (and why)

- **Not training from scratch**: You would need millions of examples and weeks on a cluster. Fine-tuning uses the language knowledge the model already has and just adjusts it to your task.
- **Not using an LLM (GPT-4, Claude, Llama)**: They are good at zero-shot labeling but slow, expensive, and not deployable offline. Use them to help label training data, not as the deployed classifier.
- **Not training Whisper**: The transcription quality is already excellent. Training Whisper would require thousands of hours of audio.

### Suggested order of operations

1. Run 5 more sessions through the pipeline with Excel/CSV input
2. Review and correct the `code` column in each CSV (10–15 min per session)
3. Once you have ~10 sessions, run `scripts/train_classifier.py`
4. Compare the trained model's moment tags against the regex tags on a held-out session
5. If accuracy > 80%, swap it in; if not, add more labeled data and retrain
