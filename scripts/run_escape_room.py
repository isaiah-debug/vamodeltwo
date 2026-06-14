"""Full pipeline run on the escape room experiment CSVs."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multi_csv_loader import load_experiment, build_explicit_graph
from src.psych_analysis import annotate_turns, speaker_profiles, emotional_contagion_matrix
from src.social_graph import compute_metrics, erdos_distances, export_html, export_metrics_csv

OUT = Path("output/escape_room_01")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ESCAPE ROOM EXPERIMENT — FULL PIPELINE")
print("=" * 60)

print("\n[1/5] Loading player CSVs...")
turns, edges = load_experiment("data/", session_id="escape_room_01")

print(f"\n[2/5] NLP annotation on {len(turns)} turns (VADER + emotion model)...")
turns = annotate_turns(turns, use_emotion_model=True, batch_size=32)

print("\n[3/5] Building explicit directed graph from receiver column...")
G = build_explicit_graph(edges)
metrics = compute_metrics(G)
profiles = speaker_profiles(turns)
contagion = emotional_contagion_matrix(turns)
dist_df = erdos_distances(G)

print("\n[4/5] Exporting outputs...")
export_html(G, OUT / "social_graph.html", metrics=metrics)
export_metrics_csv(metrics, OUT / "metrics.csv")
dist_df.to_csv(OUT / "erdos_distances.csv")

with open(OUT / "turns_annotated.json", "w", encoding="utf-8") as f:
    json.dump(turns, f, indent=2, ensure_ascii=False)
with open(OUT / "speaker_profiles.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=2, ensure_ascii=False)
with open(OUT / "edges.json", "w", encoding="utf-8") as f:
    json.dump(edges, f, indent=2, ensure_ascii=False)
with open(OUT / "contagion.json", "w", encoding="utf-8") as f:
    json.dump(contagion, f, indent=2, ensure_ascii=False)

print("\n[5/5] RESULTS")
print("=" * 60)
print("\nLEADERSHIP SCORES (eigenvector centrality = who others respond to most):")
ranked = sorted(metrics.items(), key=lambda x: x[1]["eigenvector_centrality"], reverse=True)
for rank, (spk, m) in enumerate(ranked, 1):
    print(f"  #{rank} Player {spk}: eigenvec={m['eigenvector_centrality']:.3f}  "
          f"pagerank={m['pagerank']:.3f}  dominance={m['dominance_ratio']:.2f}  "
          f"turns={m['turn_count']}")

print("\nDIRECTED COMMUNICATION MATRIX (rows=speaker, cols=receiver, values=utterance count):")
import pandas as pd
players = sorted(G.nodes)
mat = pd.DataFrame(0, index=players, columns=players)
for u, v, d in G.edges(data=True):
    mat.loc[u, v] = d["weight"]
print(mat.to_string())

print("\nSPEAKER PSYCHOLOGICAL PROFILES:")
for spk in sorted(profiles):
    p = profiles[spk]
    print(f"\n  Player {spk}:")
    print(f"    turns            : {p['turn_count']}")
    print(f"    mean sentiment   : {p['mean_sentiment']:+.3f}")
    print(f"    verbal dominance : {p['mean_verbal_dominance']:+.2f}")
    print(f"    assertive %      : {p['assertive_pct']:.0%}")
    print(f"    directive %      : {p['directive_pct']:.0%}")
    print(f"    hedging %        : {p['hedging_pct']:.0%}")
    print(f"    agreeable %      : {p['agreeable_pct']:.0%}")
    print(f"    emotion profile  : {p.get('emotion_distribution', {})}")

print("\nERDOS DISTANCES:")
print(dist_df.to_string())

print(f"\nAll outputs in: {OUT.resolve()}")
print("Open output/escape_room_01/social_graph.html in your browser.")
