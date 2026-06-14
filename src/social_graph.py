"""
Erdős-style social graph builder.

Constructs a weighted directed graph from sequential dialogue turns where:
  - Nodes  = speakers/participants
  - Edges  = A spoke → B responded  (sequential adjacency)
  - Weight = frequency of that response pair

Computes graph-theoretic metrics that map to psychological constructs:

  Metric                     Psychological construct
  ─────────────────────────  ─────────────────────────────────────────
  Eigenvector centrality     Leadership emergence
  In-degree / out-degree     Dominance ratio (high in = others direct to you)
  Betweenness centrality     Information brokerage / gatekeeper role
  PageRank                   Social influence propagation
  Reciprocity                Mutual responsiveness, rapport
  Clustering coefficient     Subgroup cohesion / coalition formation
  Erdős number (distance)    Social distance between any two speakers

Usage:
    from src.social_graph import build_graph, compute_metrics
    G = build_graph(turns)
    metrics = compute_metrics(G)
    export_html(G, "output/graph.html")   # interactive pyvis graph
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd


# ── Build ──────────────────────────────────────────────────────────────────────

def build_graph(
    turns: list[dict],
    weight_key: str = "weight",
    sentiment_key: Optional[str] = "sentiment",
) -> nx.DiGraph:
    """
    Build a directed weighted graph from a list of turn dicts.

    Sequential adjacency rule: for turns [A, B, C, ...], add edge A→B, B→C, etc.
    If turns already have a 'sentiment' field (from psych_analysis), edges are
    annotated with mean sentiment of the *response* turn.

    Parameters
    ----------
    turns        : Output of excel_parser.load_dialogue (or + psych_analysis).
    weight_key   : Name for the edge weight attribute.
    sentiment_key: Turn key to pull sentiment from (None to skip).
    """
    G = nx.DiGraph()

    speakers = sorted({t["speaker"] for t in turns})
    for spk in speakers:
        speaker_turns = [t for t in turns if t["speaker"] == spk]
        G.add_node(
            spk,
            turn_count=len(speaker_turns),
            label=spk,
        )

    for i in range(len(turns) - 1):
        src = turns[i]["speaker"]
        dst = turns[i + 1]["speaker"]
        if src == dst:
            continue  # skip self-loops (consecutive same-speaker turns)

        sentiment = None
        if sentiment_key and sentiment_key in turns[i + 1]:
            sentiment = turns[i + 1][sentiment_key]

        if G.has_edge(src, dst):
            G[src][dst][weight_key] += 1
            if sentiment is not None:
                G[src][dst]["_sentiment_sum"] += sentiment
                G[src][dst]["_sentiment_n"] += 1
        else:
            G.add_edge(
                src, dst,
                **{weight_key: 1},
                _sentiment_sum=sentiment or 0.0,
                _sentiment_n=1 if sentiment is not None else 0,
            )

    # Finalise mean sentiment per edge
    for u, v, data in G.edges(data=True):
        if data["_sentiment_n"] > 0:
            data["mean_sentiment"] = data["_sentiment_sum"] / data["_sentiment_n"]
        else:
            data["mean_sentiment"] = None
        del data["_sentiment_sum"], data["_sentiment_n"]

    return G


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(G: nx.DiGraph) -> dict[str, dict]:
    """
    Return per-node psychological metrics derived from the graph.

    Returns a dict keyed by speaker name:
        {
          "Alice": {
            "eigenvector_centrality":  0.82,   # leadership emergence
            "in_degree_centrality":    0.60,
            "out_degree_centrality":   0.40,
            "dominance_ratio":         1.50,   # in / out  (>1 = dominant)
            "betweenness_centrality":  0.33,   # gatekeeper
            "pagerank":                0.28,   # influence
            "clustering":              0.55,   # coalition
            "raw_in_degree":           6,
            "raw_out_degree":          4,
            "turn_count":              18,
          },
          ...
        }
    """
    if len(G) == 0:
        return {}

    # Eigenvector centrality (leadership proxy) — needs weight; may not converge
    # on disconnected graphs, so fall back gracefully
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G.nodes}

    in_deg  = nx.in_degree_centrality(G)
    out_deg = nx.out_degree_centrality(G)
    btwn    = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pr      = nx.pagerank(G, weight="weight")

    # Clustering on undirected projection (mutual ties)
    G_und = G.to_undirected()
    clust = nx.clustering(G_und)

    results = {}
    for node in G.nodes:
        raw_in  = G.in_degree(node, weight="weight")
        raw_out = G.out_degree(node, weight="weight")
        dom = raw_in / raw_out if raw_out > 0 else float("inf")
        results[node] = {
            "eigenvector_centrality": round(eig.get(node, 0.0), 4),
            "in_degree_centrality":   round(in_deg[node], 4),
            "out_degree_centrality":  round(out_deg[node], 4),
            "dominance_ratio":        round(dom, 4),
            "betweenness_centrality": round(btwn[node], 4),
            "pagerank":               round(pr[node], 4),
            "clustering":             round(clust.get(node, 0.0), 4),
            "raw_in_degree":          int(raw_in),
            "raw_out_degree":         int(raw_out),
            "turn_count":             G.nodes[node].get("turn_count", 0),
        }

    return results


def erdos_distances(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute all-pairs shortest-path distances on the undirected projection.
    Analogous to Erdős numbers: how many interaction steps between any two speakers.
    Returns a symmetric DataFrame (speakers × speakers).
    """
    G_und = G.to_undirected()
    nodes = sorted(G_und.nodes)
    dist = {}
    for src in nodes:
        row = {}
        for dst in nodes:
            try:
                row[dst] = nx.shortest_path_length(G_und, src, dst)
            except nx.NetworkXNoPath:
                row[dst] = float("inf")
        dist[src] = row
    return pd.DataFrame(dist, index=nodes, columns=nodes)


# ── Export ─────────────────────────────────────────────────────────────────────

def export_html(
    G: nx.DiGraph,
    out_path: str | Path = "output/graph.html",
    metrics: Optional[dict] = None,
    height: str = "750px",
) -> Path:
    """
    Export an interactive pyvis graph.  Node size = PageRank; color = dominance.
    """
    from pyvis.network import Network

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height=height, width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="white", notebook=False)

    # Node styling
    max_pr = max((metrics[n]["pagerank"] for n in G.nodes if metrics), default=1.0) or 1.0
    for node in G.nodes:
        m = (metrics or {}).get(node, {})
        pr    = m.get("pagerank", 1 / len(G))
        dom   = m.get("dominance_ratio", 1.0)
        size  = max(15, int(pr / max_pr * 60))

        # Color: warm (dominant, high in-degree) → cool (recessive)
        if dom > 1.5:
            color = "#e94560"   # red-ish
        elif dom > 1.0:
            color = "#f5a623"   # amber
        else:
            color = "#4ecdc4"   # teal

        title = "<br>".join([
            f"<b>{node}</b>",
            f"PageRank: {m.get('pagerank', 0):.3f}",
            f"Eigenvector: {m.get('eigenvector_centrality', 0):.3f}",
            f"Dominance ratio: {m.get('dominance_ratio', 0):.2f}",
            f"Betweenness: {m.get('betweenness_centrality', 0):.3f}",
            f"Turns: {m.get('turn_count', 0)}",
        ])

        net.add_node(node, label=node, size=size, color=color, title=title,
                     font={"size": 14, "color": "white"})

    # Edge styling
    max_w = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        sent = data.get("mean_sentiment")
        edge_color = "#888888"
        if sent is not None:
            edge_color = "#5cb85c" if sent > 0.05 else "#d9534f" if sent < -0.05 else "#888888"
        net.add_edge(u, v, width=max(1, w / max_w * 8),
                     color=edge_color,
                     title=f"{u} → {v}  (×{w})" + (
                         f"<br>mean sentiment: {sent:.2f}" if sent is not None else ""
                     ))

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": { "gravitationalConstant": -8000, "centralGravity": 0.3,
                       "springLength": 200, "springConstant": 0.04 }
      },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)

    net.write_html(str(out_path))
    print(f"[social_graph] interactive graph -> {out_path}")
    return out_path


def export_metrics_csv(metrics: dict, out_path: str | Path = "output/metrics.csv") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).T.to_csv(out_path)
    print(f"[social_graph] metrics CSV -> {out_path}")
    return out_path
