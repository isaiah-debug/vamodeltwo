"""
Psychological NLP analysis layer.

Annotates each dialogue turn with:
  - VADER compound sentiment score  (-1 negative … +1 positive)
  - HuggingFace emotion label       (joy/anger/sadness/fear/disgust/surprise/neutral)
  - Linguistic dominance markers    (assertive, hedging, directive, agreeable)
  - Speaker-level aggregates        (mean sentiment, emotion profile, verbal dominance)

GPU is used automatically when available (RTX 5060 via CUDA).

Usage:
    from src.psych_analysis import annotate_turns, speaker_profiles
    turns = annotate_turns(turns)          # adds fields to each turn dict
    profiles = speaker_profiles(turns)     # per-speaker aggregate stats
"""

from __future__ import annotations

import re
from typing import Optional
import warnings

# VADER — fast rule-based sentiment, works without GPU
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Transformers emotion model loaded lazily (heavy, GPU-accelerated)
_emotion_pipeline = None
_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def _get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        print(f"[psych_analysis] loading emotion model on {'GPU' if device == 0 else 'CPU'}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _emotion_pipeline = hf_pipeline(
                "text-classification",
                model=_EMOTION_MODEL,
                top_k=1,
                device=device,
                truncation=True,
                max_length=512,
            )
        print(f"[psych_analysis] emotion model ready")
    return _emotion_pipeline


# ── Dominance marker lexicons ──────────────────────────────────────────────────

_ASSERTIVE = re.compile(
    r"\b(I (will|am|think|believe|know|say)|we (should|must|need to)|"
    r"definitely|clearly|obviously|certainly|absolutely)\b",
    re.IGNORECASE,
)
_HEDGING = re.compile(
    r"\b(maybe|perhaps|I (guess|suppose|wonder)|kind of|sort of|"
    r"might|could be|possibly|not sure|I don't know)\b",
    re.IGNORECASE,
)
_DIRECTIVE = re.compile(
    r"\b(you should|you need to|let's|we have to|do it|make sure|"
    r"go ahead|proceed|start|stop|focus)\b",
    re.IGNORECASE,
)
_AGREEABLE = re.compile(
    r"\b(yes|yeah|sure|agree|absolutely|exactly|right|good point|"
    r"building on|that's true|I see)\b",
    re.IGNORECASE,
)
_INTERRUPT = re.compile(
    r"\b(but wait|actually|no no|hold on|let me|sorry but)\b",
    re.IGNORECASE,
)


def _dominance_markers(text: str) -> dict:
    return {
        "assertive":  bool(_ASSERTIVE.search(text)),
        "hedging":    bool(_HEDGING.search(text)),
        "directive":  bool(_DIRECTIVE.search(text)),
        "agreeable":  bool(_AGREEABLE.search(text)),
        "interrupting": bool(_INTERRUPT.search(text)),
    }


def _verbal_dominance_score(markers: dict) -> float:
    """Simple scalar: assertive + directive push up; hedging + agreeable push down."""
    score = 0.0
    if markers["assertive"]:   score += 1.0
    if markers["directive"]:   score += 1.5
    if markers["hedging"]:     score -= 0.8
    if markers["agreeable"]:   score -= 0.5
    if markers["interrupting"]: score += 0.5
    return round(score, 2)


# ── Turn annotation ────────────────────────────────────────────────────────────

def annotate_turns(
    turns: list[dict],
    use_emotion_model: bool = True,
    batch_size: int = 32,
) -> list[dict]:
    """
    Add NLP annotations to every turn dict (in place + return).

    Added fields per turn:
        sentiment         float   VADER compound (-1 to +1)
        sentiment_label   str     "positive" / "neutral" / "negative"
        emotion           str     top HuggingFace emotion label (or None)
        emotion_score     float   confidence of that label (or None)
        dominance_markers dict    assertive/hedging/directive/agreeable/interrupting
        verbal_dominance  float   scalar dominance score
    """
    vader = SentimentIntensityAnalyzer()

    # VADER pass (instant, CPU)
    for turn in turns:
        text = turn.get("utterance", "")
        scores = vader.polarity_scores(text)
        compound = scores["compound"]
        turn["sentiment"] = round(compound, 4)
        turn["sentiment_label"] = (
            "positive" if compound >= 0.05 else
            "negative" if compound <= -0.05 else
            "neutral"
        )
        markers = _dominance_markers(text)
        turn["dominance_markers"] = markers
        turn["verbal_dominance"] = _verbal_dominance_score(markers)

    # HuggingFace emotion pass (GPU-batched)
    if use_emotion_model:
        try:
            pipe = _get_emotion_pipeline()
            texts = [t.get("utterance", "") or "" for t in turns]
            # Batch inference — saturates GPU
            results = pipe(texts, batch_size=batch_size)
            for turn, result in zip(turns, results):
                top = result[0] if isinstance(result, list) else result
                turn["emotion"]       = top["label"].lower()
                turn["emotion_score"] = round(top["score"], 4)
        except Exception as exc:
            print(f"[psych_analysis] emotion model skipped: {exc}")
            for turn in turns:
                turn["emotion"] = None
                turn["emotion_score"] = None
    else:
        for turn in turns:
            turn["emotion"] = None
            turn["emotion_score"] = None

    return turns


# ── Speaker-level aggregates ───────────────────────────────────────────────────

def speaker_profiles(turns: list[dict]) -> dict[str, dict]:
    """
    Aggregate per-speaker statistics from annotated turns.

    Returns dict keyed by speaker name:
        {
          "Alice": {
            "mean_sentiment":       0.12,
            "sentiment_std":        0.31,
            "emotion_distribution": {"joy": 0.4, "neutral": 0.3, ...},
            "mean_verbal_dominance": 0.85,
            "assertive_pct":        0.55,
            "hedging_pct":          0.10,
            "directive_pct":        0.30,
            "agreeable_pct":        0.15,
            "interrupting_pct":     0.20,
            "turn_count":           18,
          },
          ...
        }
    """
    from collections import defaultdict
    import statistics

    grouped: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        grouped[t["speaker"]].append(t)

    profiles = {}
    for spk, spk_turns in grouped.items():
        sentiments = [t.get("sentiment", 0.0) for t in spk_turns]
        dominances = [t.get("verbal_dominance", 0.0) for t in spk_turns]
        n = len(spk_turns)

        emotions = [t.get("emotion") for t in spk_turns if t.get("emotion")]
        emotion_dist: dict[str, float] = {}
        if emotions:
            for em in set(emotions):
                emotion_dist[em] = round(emotions.count(em) / len(emotions), 3)

        def pct(key: str) -> float:
            return round(
                sum(1 for t in spk_turns if t.get("dominance_markers", {}).get(key)) / n, 3
            )

        profiles[spk] = {
            "mean_sentiment":        round(statistics.mean(sentiments), 4),
            "sentiment_std":         round(statistics.stdev(sentiments) if n > 1 else 0.0, 4),
            "emotion_distribution":  emotion_dist,
            "mean_verbal_dominance": round(statistics.mean(dominances), 4),
            "assertive_pct":         pct("assertive"),
            "hedging_pct":           pct("hedging"),
            "directive_pct":         pct("directive"),
            "agreeable_pct":         pct("agreeable"),
            "interrupting_pct":      pct("interrupting"),
            "turn_count":            n,
        }

    return profiles


# ── Emotional contagion ────────────────────────────────────────────────────────

def emotional_contagion_matrix(turns: list[dict]) -> dict:
    """
    For each directed speaker pair (A→B), compute the mean change in B's
    sentiment in the turn immediately after an emotionally charged turn by A.

    Returns a nested dict: contagion[A][B] = mean delta sentiment of B after A.
    Positive = B becomes more positive after A speaks; negative = opposite.
    """
    from collections import defaultdict

    deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for i in range(len(turns) - 1):
        a = turns[i]
        b = turns[i + 1]
        if a["speaker"] == b["speaker"]:
            continue
        a_sent = a.get("sentiment")
        b_sent = b.get("sentiment")
        if a_sent is None or b_sent is None:
            continue
        if abs(a_sent) < 0.05:
            continue  # only count emotionally charged A turns
        delta = b_sent - a_sent
        deltas[a["speaker"]][b["speaker"]].append(delta)

    result: dict[str, dict[str, Optional[float]]] = {}
    for a_spk, targets in deltas.items():
        result[a_spk] = {}
        for b_spk, ds in targets.items():
            result[a_spk][b_spk] = round(sum(ds) / len(ds), 4) if ds else None

    return result
