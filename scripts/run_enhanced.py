"""Run full pipeline + enhanced visualization on the escape room CSVs."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multi_csv_loader import load_experiment, build_explicit_graph
from src.psych_analysis import annotate_turns, speaker_profiles, emotional_contagion_matrix
from src.social_graph import compute_metrics, erdos_distances
from src.leadership_assessment import full_assessment
from src.graph_enhanced import export_enhanced_html

OUT = Path("output/escape_room_01")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ESCAPE ROOM — ENHANCED LEADERSHIP ANALYSIS")
print("=" * 60)

print("\n[1/5] Loading CSVs...")
turns, edges = load_experiment("data/", session_id="escape_room_01")

print(f"\n[2/5] NLP + emotion annotation ({len(turns)} turns, GPU)...")
turns = annotate_turns(turns, use_emotion_model=True, batch_size=32)

print("\n[3/5] Social graph + network metrics...")
G = build_explicit_graph(edges)
metrics   = compute_metrics(G)
profiles  = speaker_profiles(turns)

print("\n[4/5] Leadership assessment (Bass & Avolio + IPA + Hollander + Bavelas)...")
ld_scores, moments, composites, styles = full_assessment(turns, metrics, profiles)

print(f"  Found {len(moments)} leadership moments")
for spk in sorted(styles):
    print(f"  Player {spk}: {styles[spk]}  (composite={composites[spk]:.3f})")

print("\n[5/5] Generating enhanced visualization...")
out = export_enhanced_html(
    G=G,
    turns=turns,
    metrics=metrics,
    ld_scores=ld_scores,
    moments=moments,
    composites=composites,
    styles=styles,
    profiles=profiles,
    out_path=OUT / "enhanced_analysis.html",
)

# Save analysis JSON
with open(OUT / "leadership_scores.json", "w", encoding="utf-8") as f:
    # Remove _raw for cleaner output
    clean = {k: {kk: vv for kk, vv in v.items() if kk != "_raw"}
             for k, v in ld_scores.items()}
    json.dump(clean, f, indent=2)
with open(OUT / "leadership_moments.json", "w", encoding="utf-8") as f:
    json.dump(moments, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("LEADERSHIP SUMMARY")
print("=" * 60)
ranked = sorted(composites.items(), key=lambda x: x[1], reverse=True)
for rank, (spk, score) in enumerate(ranked, 1):
    sc = ld_scores[spk]
    print(f"\n  #{rank}  Player {spk}  [{styles[spk]}]")
    print(f"       Composite emergent score : {score:.3f}")
    print(f"       Eigenvec centrality      : {metrics.get(spk,{}).get('eigenvector_centrality',0):.3f}")
    print(f"       Talk share               : {sc['talk_share']:.1%}")
    print(f"       Transformational         : {sc['transformational']:.3f}")
    print(f"       Transactional            : {sc['transactional']:.3f}")
    print(f"       Laissez-faire            : {sc['laissez_faire']:.3f}")
    print(f"       Insight rate             : {sc['insight_rate']:.3f}")
    print(f"       IPA Task                 : {sc['ipa_task']:.3f}")
    print(f"       IPA Socio-positive       : {sc['ipa_socio_pos']:.3f}")

spk_moms = {}
from collections import Counter
for m in moments:
    spk_moms.setdefault(m["speaker"], []).append(m)
print(f"\nLEADERSHIP MOMENTS ({len(moments)} total):")
for spk in sorted(spk_moms):
    tags = Counter(t for m in spk_moms[spk] for t in m["tags"])
    print(f"  Player {spk}: {dict(tags)}")

print(f"\nOpen: {(OUT / 'enhanced_analysis.html').resolve()}")
