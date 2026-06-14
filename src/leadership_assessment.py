"""
Leadership Assessment Framework — econ-psych literature synthesis.

Core frameworks drawn from:

  Bass & Avolio (1994) — Full Range Leadership Model
    Transformational: inspirational motivation, intellectual stimulation,
                      individualized consideration, idealised influence
    Transactional:    contingent reward, management by exception (active)
    Laissez-faire:    passive avoidance, non-intervention

  Bales (1950, 1970) — Interaction Process Analysis (IPA)
    Task-oriented acts:          gives orientation / opinion / suggestion
    Socio-emotional positive:    shows solidarity, releases tension, agrees
    Socio-emotional negative:    disagrees, shows tension, shows antagonism

  Hollander (1958) — Idiosyncrasy Credits / Emergent Leadership
    Leaders earn influence through early conformity + demonstrated competence.
    Proxy here: information contribution rate early in session.

  Bavelas (1950), Freeman (1979) — Network Centrality → Leadership
    Betweenness, eigenvector centrality predict emergent influence.

  Mast (2002) — Speaking Time as Dominance
    Speaking-time proportion is one of the strongest correlates of
    perceived leadership.

  Anderson & Kilduff (2009) — Dominant Personalities
    Dominant speakers claim status through confident, assertive framing,
    not just volume.

  Willer, Rogalin, Conlon & Wojnowicz (2012) — Legitimacy & Deference
    Deference received (in-degree) signals perceived competence.

  Yukl (2012) — Flexible Leadership Theory
    Effective leaders balance task, relation, and change behaviours.
    We measure each axis independently.

OUTPUT: per-speaker leadership profile with composite scores and style label.
"""

from __future__ import annotations

import re
from typing import Optional

# ── Linguistic pattern banks ───────────────────────────────────────────────────

# --- Transformational markers (Bass & Avolio 1994, Bono & Judge 2004) ---
_TRANSFORM_INSPIRE = re.compile(
    r"\b(let'?s go|we can|we got (it|this)|come on|we'?re going to|"
    r"we'?ll|imagine|what if we|picture this|the idea is|think about)\b",
    re.I,
)
_TRANSFORM_INTELLECTUAL = re.compile(
    r"\b(what if|maybe we could|another way|have you considered|"
    r"different approach|think about it|could also|alternative|"
    r"wait what about|hold on|I wonder)\b",
    re.I,
)
_TRANSFORM_CONSIDERATE = re.compile(
    r"\b(good (point|idea|call|thinking)|nice|great|exactly right|"
    r"building on (that|what)|you're right|I agree with|that works|"
    r"that's true|I see what you mean)\b",
    re.I,
)
_TRANSFORM_COLLECTIVE = re.compile(r"\bwe\b", re.I)   # collective framing

# --- Transactional markers (Burns 1978, Bass 1985) ---
_TRANS_DIRECTIVE = re.compile(
    r"\b(you should|you need to|go (ahead|check|look|try)|"
    r"do (that|this|it)|make sure|don'?t forget|focus on|"
    r"let'?s (split|try|check|go|do|focus|move))\b",
    re.I,
)
_TRANS_TASK = re.compile(
    r"\b(the (lock|code|key|clue|puzzle|numbers?|colors?|weight|scale|"
    r"combination|solution|answer|document|page|note|vial|serum|jar))\b",
    re.I,
)
_TRANS_COORDINATE = re.compile(
    r"\b(did (you|we) (try|find|check|get|do)|have (you|we)|"
    r"did (it|that) work|any luck|what (did|do) you (find|get|see))\b",
    re.I,
)

# --- Laissez-faire / passive markers (Bass 1994) ---
_LF_PASSIVE = re.compile(
    r"^(ok|okay|yeah|yes|sure|mhm|uh huh|nope|yep|right|alright|"
    r"ah|oh|hmm|um|uh)\.?$",
    re.I,
)
_LF_HEDGE = re.compile(
    r"\b(I (guess|suppose|dunno|don'?t know)|maybe|probably|"
    r"I'?m not sure|kind of|sort of|I think so|could be|might be|"
    r"not sure if)\b",
    re.I,
)

# --- IPA: task vs socio-emotional (Bales 1950) ---
_IPA_TASK_ORIENT = re.compile(
    r"\b(what (is|are|does|did|do)|how (do|does|did|many|much)|"
    r"which (one|way)|where (is|are)|when (did|does)|"
    r"can (we|you|I)|should (we|I)|let'?s (try|check|see))\b",
    re.I,
)
_IPA_SOCIO_POS = re.compile(
    r"\b(haha|lol|nice|awesome|yay|ahh|woo|great|cool|"
    r"oh (no|wow|nice|interesting)|I (love|like) (that|this|it))\b",
    re.I,
)
_IPA_SOCIO_NEG = re.compile(
    r"\b(no (no|way|that'?s)|that'?s (wrong|not|bad)|"
    r"I (disagree|don'?t think|doubt)|wait (no|that'?s))\b",
    re.I,
)

# --- Idiosyncrasy credit proxy (Hollander 1958) ---
# High-value contributions: sharing a key insight or solution
_INSIGHT = re.compile(
    r"\b(I (found|got|figured|think I|see)|oh (wait|I see|I get it)|"
    r"that'?s (it|the|why)|we need to|the answer (is|might)|"
    r"what if it'?s|maybe (it'?s|the)|so (it'?s|the|we))\b",
    re.I,
)

# ── Scoring ────────────────────────────────────────────────────────────────────

def _pct(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def score_turns(turns: list[dict]) -> dict[str, dict]:
    """
    Compute per-speaker leadership scores from annotated turns.

    Returns dict keyed by speaker:
    {
      "A": {
        # Bass & Avolio dimensions (0–1 proportions)
        "transformational":   0.32,
        "transactional":      0.45,
        "laissez_faire":      0.10,
        # IPA dimensions
        "ipa_task":           0.60,
        "ipa_socio_pos":      0.15,
        "ipa_socio_neg":      0.04,
        # Hollander idiosyncrasy credit proxy
        "insight_rate":       0.18,
        # Talk-time (Mast 2002)
        "talk_share":         0.32,   # fraction of all turns
        "turn_count":         162,
        # Composite emergent leadership score (network + behavioural)
        "emergent_score":     0.71,
        # Style label
        "leadership_style":   "Transactional–Organiser",
        # Raw counts for radar
        "_raw": { ... }
      },
      ...
    }
    """
    from collections import defaultdict

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        by_speaker[t["speaker"]].append(t)

    total_turns = len(turns)
    results = {}

    for spk, spk_turns in by_speaker.items():
        n = len(spk_turns)
        texts = [t.get("utterance", "") for t in spk_turns]

        def count(pat):
            return sum(1 for tx in texts if pat.search(tx))

        tf_inspire  = count(_TRANSFORM_INSPIRE)
        tf_intel    = count(_TRANSFORM_INTELLECTUAL)
        tf_consid   = count(_TRANSFORM_CONSIDERATE)
        tf_collect  = sum(len(_TRANSFORM_COLLECTIVE.findall(tx)) for tx in texts)
        transformational = _pct(tf_inspire + tf_intel + tf_consid, n)

        tr_direct   = count(_TRANS_DIRECTIVE)
        tr_task     = count(_TRANS_TASK)
        tr_coord    = count(_TRANS_COORDINATE)
        transactional = _pct(tr_direct + tr_task + tr_coord, n)

        lf_passive  = count(_LF_PASSIVE)
        lf_hedge    = count(_LF_HEDGE)
        laissez_faire = _pct(lf_passive + lf_hedge, n)

        ipa_task    = count(_IPA_TASK_ORIENT)
        ipa_spos    = count(_IPA_SOCIO_POS)
        ipa_sneg    = count(_IPA_SOCIO_NEG)

        insights    = count(_INSIGHT)

        # Talk share (Mast 2002) — raw fraction of all session turns
        talk_share  = round(n / total_turns, 4)

        results[spk] = {
            "transformational": transformational,
            "transactional":    transactional,
            "laissez_faire":    laissez_faire,
            "ipa_task":         _pct(ipa_task, n),
            "ipa_socio_pos":    _pct(ipa_spos, n),
            "ipa_socio_neg":    _pct(ipa_sneg, n),
            "insight_rate":     _pct(insights, n),
            "talk_share":       talk_share,
            "turn_count":       n,
            "_raw": {
                "tf_inspire": tf_inspire, "tf_intel": tf_intel,
                "tf_consid": tf_consid, "tf_collective_we": tf_collect,
                "tr_direct": tr_direct, "tr_task": tr_task, "tr_coord": tr_coord,
                "lf_passive": lf_passive, "lf_hedge": lf_hedge,
                "ipa_task": ipa_task, "ipa_socio_pos": ipa_spos, "ipa_socio_neg": ipa_sneg,
                "insights": insights,
            },
        }

    return results


def classify_style(scores: dict, net_metrics: Optional[dict] = None) -> str:
    """
    Map Bass & Avolio + network metrics to a qualitative leadership style label.

    Styles (Yukl 2012 integration):
      Transformational–Visionary : high tf, moderate tr, central network
      Transactional–Organiser    : high tr, moderate tf, directive
      Facilitative–Connector     : moderate tf+tr, high betweenness
      Analytical–Epistemic       : high ipa_task, high insight_rate, low socio
      Social–Emotional           : high socio_pos, low directive
      Laissez-faire–Follower     : high lf, low tf+tr
    """
    tf = scores["transformational"]
    tr = scores["transactional"]
    lf = scores["laissez_faire"]
    ipa_t = scores["ipa_task"]
    ipa_sp = scores["ipa_socio_pos"]
    ins = scores["insight_rate"]
    btw = (net_metrics or {}).get("betweenness_centrality", 0)
    eig = (net_metrics or {}).get("eigenvector_centrality", 0)

    dominant_dim = max(tf, tr, lf)

    if lf >= 0.25 and lf >= tf and lf >= tr:
        return "Laissez-faire / Follower"
    if tf >= 0.18 and eig > 0.5:
        return "Transformational / Visionary"
    if tr >= 0.30 and tr > tf:
        return "Transactional / Organiser"
    if btw > 0.28:
        return "Facilitative / Connector"
    if ipa_t >= 0.35 and ins >= 0.15:
        return "Analytical / Epistemic"
    if ipa_sp >= 0.10 and tf > tr:
        return "Social-Emotional / Supporter"
    if tf > tr:
        return "Transformational / Collaborative"
    return "Transactional / Task-focused"


def identify_leadership_moments(turns: list[dict]) -> list[dict]:
    """
    Tag individual turns as 'leadership moments' using the following criteria:

    Type                    Criterion
    ──────────────────────  ─────────────────────────────────────────────────
    Directive               Explicit instruction to another named player
    Insight / Solution      Proposes or identifies a solution to the puzzle
    Consensus-building      Solicits input from the group; synthesises views
    Emotional regulation    Positive affirmation after a negative moment
    Coordination            Assigns roles / re-coordinates the group
    Breakthrough            Immediately followed by group agreement / success
    """
    _DIRECTIVE_PAT  = re.compile(r"\b(go|try|check|look|do|let'?s|you should|make sure|focus)\b", re.I)
    _SOLUTION_PAT   = re.compile(r"\b(the answer|I (think|got|figured)|so (it'?s|the)|that'?s (it|the)|found it|maybe it'?s|what if (it'?s|we try))\b", re.I)
    _CONSENSUS_PAT  = re.compile(r"\b(what do (you|everyone) (think|say)|does (everyone|anyone)|any (thoughts|ideas|luck)|how (about|do) we|agree\??)\b", re.I)
    _AFFIRM_PAT     = re.compile(r"\b(good (job|point|idea|call)|nice|great|we got (it|this)|aye|yay|ahh we got)\b", re.I)
    _COORD_PAT      = re.compile(r"\b(split (up)?|you (take|handle|do)|I'?ll (take|do|handle)|who (has|wants|can)|let'?s (divide|assign|coordinate))\b", re.I)

    prev_neg = False
    moments = []

    for i, turn in enumerate(turns):
        text = turn.get("utterance", "")
        spk  = turn["speaker"]
        tags = []

        if _DIRECTIVE_PAT.search(text) and turn.get("receiver_raw", "").upper() in ("A","B","C","D"):
            tags.append("Directive")
        if _SOLUTION_PAT.search(text):
            tags.append("Insight/Solution")
        if _CONSENSUS_PAT.search(text):
            tags.append("Consensus-building")
        if prev_neg and _AFFIRM_PAT.search(text):
            tags.append("Emotional-regulation")
        if _COORD_PAT.search(text):
            tags.append("Coordination")
        # Breakthrough: check if next 2 turns are agreement
        if i < len(turns) - 2:
            next_texts = [turns[i+j].get("utterance","") for j in (1,2)]
            if any(_AFFIRM_PAT.search(t) for t in next_texts):
                if _SOLUTION_PAT.search(text):
                    tags.append("Breakthrough")

        if tags:
            moments.append({
                "turn":      turn["turn"],
                "speaker":   spk,
                "timestamp": turn.get("timestamp", ""),
                "start_s":   turn.get("start_s", 0),
                "utterance": text,
                "tags":      tags,
            })

        # Track negative sentiment for emotional regulation detection
        prev_neg = turn.get("sentiment", 0) < -0.1

    return moments


def composite_emergent_score(
    ld_scores: dict,
    net_metrics: dict,
    profiles: dict,
) -> dict[str, float]:
    """
    Composite emergent leadership score blending:
      40%  Network influence  (eigenvector centrality — Bavelas 1950)
      20%  Talk share         (Mast 2002)
      15%  Transformational   (Bass 1985)
      15%  Transactional      (Burns 1978)
      10%  Insight rate       (Hollander 1958)

    Returns dict of {speaker: score (0–1)}.
    """
    speakers = list(ld_scores.keys())

    def normalise(values):
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.5] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    eig_raw   = [net_metrics.get(s, {}).get("eigenvector_centrality", 0) for s in speakers]
    talk_raw  = [ld_scores[s]["talk_share"]        for s in speakers]
    tf_raw    = [ld_scores[s]["transformational"]  for s in speakers]
    tr_raw    = [ld_scores[s]["transactional"]     for s in speakers]
    ins_raw   = [ld_scores[s]["insight_rate"]      for s in speakers]

    eig_n  = normalise(eig_raw)
    talk_n = normalise(talk_raw)
    tf_n   = normalise(tf_raw)
    tr_n   = normalise(tr_raw)
    ins_n  = normalise(ins_raw)

    scores = {}
    for i, spk in enumerate(speakers):
        scores[spk] = round(
            0.40 * eig_n[i]  +
            0.20 * talk_n[i] +
            0.15 * tf_n[i]   +
            0.15 * tr_n[i]   +
            0.10 * ins_n[i],
            4,
        )
    return scores


def full_assessment(turns, net_metrics, profiles):
    """
    Run the complete leadership assessment pipeline.
    Returns (ld_scores, moments, composites, styles).
    """
    ld_scores  = score_turns(turns)
    moments    = identify_leadership_moments(turns)
    composites = composite_emergent_score(ld_scores, net_metrics, profiles)

    styles = {}
    for spk, sc in ld_scores.items():
        sc["composite_emergent"] = composites.get(spk, 0)
        sc["leadership_style"]   = classify_style(sc, net_metrics.get(spk))
        styles[spk] = sc["leadership_style"]

    return ld_scores, moments, composites, styles
