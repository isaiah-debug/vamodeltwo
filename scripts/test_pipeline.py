"""Quick end-to-end pipeline test. Run: python scripts/test_pipeline.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=== STEP 1: Parse Excel ===")
from src.excel_parser import load_dialogue
turns = load_dialogue(
    "data/escape_room_dialogue_EXAMPLE.xlsx",
    sheet="Dialogue Coding",
    session_id="escape_room_01",
)
speakers = sorted(set(t["speaker"] for t in turns))
print(f"Loaded {len(turns)} turns, speakers: {speakers}")
for t in turns[:3]:
    print(f"  turn {t['turn']}  [{t['speaker']}]  {t['utterance'][:55]}")

print()
print("=== STEP 2: NLP Annotation (VADER, no GPU model for speed) ===")
from src.psych_analysis import annotate_turns, speaker_profiles, emotional_contagion_matrix
turns = annotate_turns(turns, use_emotion_model=False)
for t in turns[:3]:
    print(f"  [{t['speaker']}] sentiment={t['sentiment']:+.2f}  verbal_dom={t['verbal_dominance']:+.1f}  {t['utterance'][:45]}")

print()
print("=== STEP 3: Social Graph ===")
from src.social_graph import build_graph, compute_metrics, erdos_distances
G = build_graph(turns)
metrics = compute_metrics(G)
print(f"  Nodes : {list(G.nodes)}")
print(f"  Edges : {[(u, v, d['weight']) for u, v, d in G.edges(data=True)]}")

print()
print("=== STEP 4: Psychological Metrics ===")
for spk, m in sorted(metrics.items()):
    print(
        f"  Player {spk}: "
        f"eigenvec={m['eigenvector_centrality']:.3f}  "
        f"pagerank={m['pagerank']:.3f}  "
        f"dominance={m['dominance_ratio']:.2f}  "
        f"turns={m['turn_count']}"
    )

print()
print("=== STEP 5: Speaker Profiles ===")
profiles = speaker_profiles(turns)
for spk, p in sorted(profiles.items()):
    print(
        f"  Player {spk}: "
        f"mean_sentiment={p['mean_sentiment']:+.3f}  "
        f"verbal_dom={p['mean_verbal_dominance']:+.2f}  "
        f"assertive={p['assertive_pct']:.0%}  "
        f"directive={p['directive_pct']:.0%}"
    )

print()
print("=== STEP 6: Erdos Distances ===")
dist = erdos_distances(G)
print(dist.to_string())

print()
print("=== STEP 7: Export Graph HTML ===")
from src.social_graph import export_html
Path("output").mkdir(exist_ok=True)
export_html(G, "output/test_graph.html", metrics=metrics)
print("  Written: output/test_graph.html")

print()
print("ALL STEPS PASSED — pipeline is working correctly.")
