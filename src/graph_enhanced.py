"""
Enhanced visualization: cube-layout social graph + discrete interaction timeline
+ leadership key + radar charts. Outputs a single self-contained HTML file.

Uses Plotly only (no pyvis dependency) for full layout control.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette ─────────────────────────────────────────────────────────────
PLAYER_COLORS = {
    "A": "#e94560",   # crimson
    "B": "#0f9b8e",   # teal
    "C": "#f5a623",   # amber
    "D": "#7b5ea7",   # violet
}

STYLE_COLORS = {
    "Transformational / Visionary":     "#e94560",
    "Transformational / Collaborative": "#f07c5a",
    "Transactional / Organiser":        "#f5a623",
    "Transactional / Task-focused":     "#d4b200",
    "Facilitative / Connector":         "#0f9b8e",
    "Analytical / Epistemic":           "#3a9bd5",
    "Social-Emotional / Supporter":     "#7b5ea7",
    "Laissez-faire / Follower":         "#888888",
}

MOMENT_COLORS = {
    "Directive":            "#e94560",
    "Insight/Solution":     "#00d4aa",
    "Consensus-building":   "#f5a623",
    "Emotional-regulation": "#7b5ea7",
    "Coordination":         "#3a9bd5",
    "Breakthrough":         "#ffd700",
}

# Fixed cube-corner positions for 4 players (in 2D projected cube)
CUBE_POS = {
    "A": (-0.85,  0.72),
    "B": ( 0.85,  0.72),
    "C": (-0.85, -0.72),
    "D": ( 0.85, -0.72),
}


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _edge_midpoint(src: str, dst: str, offset_factor: float = 0.0):
    x0, y0 = CUBE_POS[src]
    x1, y1 = CUBE_POS[dst]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    # Perp offset so bidirectional edges don't overlap
    dx, dy = x1 - x0, y1 - y0
    length = math.sqrt(dx**2 + dy**2) or 1
    return mx + offset_factor * (-dy / length), my + offset_factor * (dx / length)


def _bezier(x0, y0, cx, cy, x1, y1, n=40):
    """Quadratic bezier curve points."""
    xs, ys = [], []
    for i in range(n + 1):
        t = i / n
        x = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
        y = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
        xs.append(x); ys.append(y)
    return xs, ys


# ── Main export function ───────────────────────────────────────────────────────

def export_enhanced_html(
    G: nx.DiGraph,
    turns: list[dict],
    metrics: dict,
    ld_scores: dict,
    moments: list[dict],
    composites: dict[str, float],
    styles: dict[str, str],
    profiles: dict,
    out_path: str | Path = "output/enhanced_analysis.html",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    players = sorted(CUBE_POS.keys())
    session_duration = max((t.get("start_s", 0) or 0) for t in turns)

    # ── Figure layout: 3 rows ────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"colspan": 2, "rowspan": 2, "type": "xy"}, None,
             {"type": "polar"}],
            [None, None,
             {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Social Graph — Who Responds to Whom",
            "Leadership Radar",
            "",                          # communication volume (row2 col3)
            "Discrete Interaction Timeline",
            "Leadership Moment Map",
            "Composite Emergent Score",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
        row_heights=[0.38, 0.12, 0.36],
        column_widths=[0.38, 0.32, 0.30],
    )

    # ════════════════════════════════════════════════════════════════════════
    # ROW 1-2, COL 1-2: Social graph (cube layout)
    # ════════════════════════════════════════════════════════════════════════
    max_w = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)

    # Edges — curved bezier per direction pair
    for u, v, data in G.edges(data=True):
        if u not in CUBE_POS or v not in CUBE_POS:
            continue
        w = data.get("weight", 1)
        x0, y0 = CUBE_POS[u]
        x1, y1 = CUBE_POS[v]
        # Control point offset: clockwise curve
        cx, cy = _edge_midpoint(u, v, 0.22)
        bx, by = _bezier(x0, y0, cx, cy, x1, y1, n=50)

        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(
                color=PLAYER_COLORS.get(u, "#888"),
                width=max(1.5, w / max_w * 10),
            ),
            opacity=0.65,
            hoverinfo="text",
            hovertext=f"{u} → {v}  ({w} utterances)",
            showlegend=False,
        ), row=1, col=1)

        # Arrowhead: place a marker 85% along the curve
        ai = int(0.85 * len(bx))
        fig.add_trace(go.Scatter(
            x=[bx[ai]], y=[by[ai]], mode="markers",
            marker=dict(
                symbol="triangle-right",
                size=10,
                color=PLAYER_COLORS.get(u, "#888"),
                angle=math.degrees(math.atan2(by[ai]-by[ai-1], bx[ai]-bx[ai-1])),
            ),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # Individual interaction dots scattered along edges
    turn_df = pd.DataFrame(turns)
    turn_df["start_s"] = pd.to_numeric(turn_df["start_s"], errors="coerce").fillna(0)
    t_max = turn_df["start_s"].max() or 1

    for _, row in turn_df.iterrows():
        src = str(row["speaker"])
        rec_raw = str(row.get("receiver_raw", ""))
        # Determine primary receiver
        if rec_raw.upper() in ("ALL", ""):
            continue  # skip broadcast for individual dots
        rec = rec_raw.split(" and ")[0].split("/")[0].strip().upper()
        if rec not in CUBE_POS or src not in CUBE_POS or rec == src:
            continue

        t_frac = float(row["start_s"]) / t_max
        cx_mid, cy_mid = _edge_midpoint(src, rec, 0.22)
        x0, y0 = CUBE_POS[src]
        x1, y1 = CUBE_POS[rec]
        # Point on bezier at t_frac
        def bez_pt(t):
            return (
                (1-t)**2 * x0 + 2*(1-t)*t * cx_mid + t**2 * x1,
                (1-t)**2 * y0 + 2*(1-t)*t * cy_mid + t**2 * y1,
            )
        px, py = bez_pt(t_frac)

        sent = float(row.get("sentiment", 0) or 0)
        dot_color = ("#00d4aa" if sent > 0.05 else
                     "#e94560" if sent < -0.05 else "#cccccc")

        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers",
            marker=dict(size=5, color=dot_color, opacity=0.55,
                        line=dict(width=0)),
            hoverinfo="text",
            hovertext=(f"[{src}→{rec}]  t={row.get('timestamp','?')}<br>"
                       f"{str(row.get('utterance',''))[:80]}<br>"
                       f"sentiment={sent:+.2f}"),
            showlegend=False,
        ), row=1, col=1)

    # Node markers (large, styled)
    for spk in players:
        if spk not in CUBE_POS:
            continue
        x, y = CUBE_POS[spk]
        m = metrics.get(spk, {})
        style = styles.get(spk, "")
        comp  = composites.get(spk, 0)
        prof  = profiles.get(spk, {})

        node_size = 28 + int(comp * 35)

        tooltip = (
            f"<b>Player {spk}</b><br>"
            f"Style: {style}<br>"
            f"Emergent score: {comp:.2f}<br>"
            f"Eigenvec centrality: {m.get('eigenvector_centrality', 0):.3f}<br>"
            f"Dominance ratio: {m.get('dominance_ratio', 0):.2f}<br>"
            f"PageRank: {m.get('pagerank', 0):.3f}<br>"
            f"Mean sentiment: {prof.get('mean_sentiment', 0):+.3f}<br>"
            f"Turns: {ld_scores.get(spk, {}).get('turn_count', 0)}"
        )

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(
                size=node_size,
                color=PLAYER_COLORS[spk],
                line=dict(width=3, color="white"),
                symbol="circle",
            ),
            text=[f"<b>{spk}</b>"],
            textposition="middle center",
            textfont=dict(size=16, color="white", family="Arial Black"),
            hoverinfo="text",
            hovertext=tooltip,
            showlegend=False,
            name=f"Player {spk}",
        ), row=1, col=1)

        # Style label below node
        fig.add_trace(go.Scatter(
            x=[x], y=[y - 0.12], mode="text",
            text=[f"<i>{style}</i>"],
            textfont=dict(size=9, color=PLAYER_COLORS[spk]),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # Leadership moment stars on the graph
    moment_plotted = set()
    for mom in moments:
        src = mom["speaker"]
        if src not in CUBE_POS or src in moment_plotted:
            continue
        x, y = CUBE_POS[src]
        tag = mom["tags"][0]
        fig.add_trace(go.Scatter(
            x=[x + 0.11], y=[y + 0.11], mode="markers",
            marker=dict(
                symbol="star",
                size=14,
                color=MOMENT_COLORS.get(tag, "#ffd700"),
                line=dict(width=1, color="white"),
            ),
            hoverinfo="text",
            hovertext=f"Leadership moment [{tag}]<br>{mom['utterance'][:80]}",
            showlegend=False,
        ), row=1, col=1)
        moment_plotted.add(src)

    fig.update_xaxes(range=[-1.4, 1.4], showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)

    # ════════════════════════════════════════════════════════════════════════
    # ROW 1, COL 3: Leadership Radar (Bass & Avolio dimensions)
    # ════════════════════════════════════════════════════════════════════════
    radar_cats = [
        "Transformational", "Transactional",
        "IPA Task", "Socio-Positive", "Insight Rate", "Talk Share",
    ]
    for spk in players:
        sc = ld_scores.get(spk, {})
        vals = [
            sc.get("transformational", 0) * 3,
            sc.get("transactional", 0) * 3,
            sc.get("ipa_task", 0) * 3,
            sc.get("ipa_socio_pos", 0) * 3,
            sc.get("insight_rate", 0) * 3,
            sc.get("talk_share", 0),
        ]
        # Close the radar polygon
        vals_closed = vals + [vals[0]]
        cats_closed = radar_cats + [radar_cats[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself",
            name=f"Player {spk}",
            line=dict(color=PLAYER_COLORS[spk], width=2),
            fillcolor=PLAYER_COLORS[spk],
            opacity=0.25,
        ), row=1, col=3)

    fig.update_polars(
        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
        angularaxis=dict(tickfont=dict(size=9, color="white")),
        bgcolor="rgba(20,20,40,0.0)",
        row=1, col=3,
    )

    # ════════════════════════════════════════════════════════════════════════
    # ROW 2, COL 3: Communication volume bars
    # ════════════════════════════════════════════════════════════════════════
    fig.add_trace(go.Bar(
        x=players,
        y=[ld_scores.get(p, {}).get("turn_count", 0) for p in players],
        marker_color=[PLAYER_COLORS[p] for p in players],
        text=[str(ld_scores.get(p, {}).get("turn_count", 0)) for p in players],
        textposition="auto",
        showlegend=False,
        hovertemplate="%{x}: %{y} turns<extra></extra>",
    ), row=2, col=3)
    fig.update_xaxes(title_text="Player", row=2, col=3,
                     tickfont=dict(color="white"))
    fig.update_yaxes(title_text="Turns", row=2, col=3,
                     tickfont=dict(color="white"))

    # ════════════════════════════════════════════════════════════════════════
    # ROW 3, COL 1: Discrete interaction timeline
    # ════════════════════════════════════════════════════════════════════════
    y_pos = {"A": 3, "B": 2, "C": 1, "D": 0}

    for spk in players:
        spk_turns = [t for t in turns if t["speaker"] == spk and t.get("start_s") is not None]
        xs = [float(t["start_s"]) for t in spk_turns]
        ys = [y_pos[spk]] * len(xs)
        sents = [float(t.get("sentiment", 0) or 0) for t in spk_turns]
        emotions = [t.get("emotion", "") or "" for t in spk_turns]
        texts_hover = [
            f"[{spk}] t={t.get('timestamp','?')}<br>{t.get('utterance','')[:70]}"
            f"<br>sentiment={s:+.2f} emotion={e}"
            for t, s, e in zip(spk_turns, sents, emotions)
        ]

        # Colour each dot by sentiment
        dot_colors = [
            "#00d4aa" if s > 0.05 else "#e94560" if s < -0.05 else PLAYER_COLORS[spk]
            for s in sents
        ]
        dot_sizes = [7 + abs(s) * 10 for s in sents]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(size=dot_sizes, color=dot_colors, opacity=0.8,
                        line=dict(width=0.5, color="white")),
            hoverinfo="text",
            hovertext=texts_hover,
            showlegend=True,
            name=f"Player {spk}",
            legendgroup=f"p{spk}",
        ), row=3, col=1)

    # Leadership moments on timeline
    for mom in moments:
        spk = mom["speaker"]
        if spk not in y_pos:
            continue
        t_s = mom.get("start_s", 0) or 0
        tag = mom["tags"][0]
        fig.add_trace(go.Scatter(
            x=[float(t_s)], y=[y_pos[spk] + 0.45],
            mode="markers+text",
            marker=dict(symbol="star", size=11,
                        color=MOMENT_COLORS.get(tag, "#ffd700"),
                        line=dict(width=1, color="white")),
            text=["★"],
            textposition="middle center",
            textfont=dict(size=7),
            hoverinfo="text",
            hovertext=f"[{tag}]<br>{mom['utterance'][:70]}",
            showlegend=False,
        ), row=3, col=1)

    fig.update_xaxes(
        title_text="Time (seconds)", row=3, col=1,
        tickfont=dict(color="white"),
        gridcolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=["D", "C", "B", "A"],
        row=3, col=1,
        tickfont=dict(color="white"),
        gridcolor="rgba(255,255,255,0.05)",
    )

    # ════════════════════════════════════════════════════════════════════════
    # ROW 3, COL 2: Leadership moment frequency bars
    # ════════════════════════════════════════════════════════════════════════
    from collections import Counter
    for spk in players:
        spk_moms = [m for m in moments if m["speaker"] == spk]
        tag_counts = Counter(tag for m in spk_moms for tag in m["tags"])
        all_tags = list(MOMENT_COLORS.keys())
        fig.add_trace(go.Bar(
            name=f"Player {spk}",
            x=all_tags,
            y=[tag_counts.get(t, 0) for t in all_tags],
            marker_color=PLAYER_COLORS[spk],
            legendgroup=f"p{spk}",
            showlegend=False,
            hovertemplate=f"Player {spk}<br>%{{x}}: %{{y}}<extra></extra>",
        ), row=3, col=2)
    fig.update_xaxes(tickfont=dict(size=8, color="white"),
                     tickangle=30, row=3, col=2)
    fig.update_yaxes(title_text="Count", tickfont=dict(color="white"), row=3, col=2)

    # ════════════════════════════════════════════════════════════════════════
    # ROW 3, COL 3: Composite emergent leadership score
    # ════════════════════════════════════════════════════════════════════════
    sorted_players = sorted(composites.keys(), key=lambda p: composites[p], reverse=True)
    fig.add_trace(go.Bar(
        x=sorted_players,
        y=[composites[p] for p in sorted_players],
        marker_color=[PLAYER_COLORS.get(p, "#888") for p in sorted_players],
        text=[f"{composites[p]:.2f}" for p in sorted_players],
        textposition="auto",
        textfont=dict(color="white"),
        showlegend=False,
        hovertemplate="Player %{x}<br>Emergent score: %{y:.3f}<extra></extra>",
    ), row=3, col=3)
    fig.update_xaxes(title_text="Player", row=3, col=3, tickfont=dict(color="white"))
    fig.update_yaxes(title_text="Score (0–1)", row=3, col=3, range=[0, 1.05],
                     tickfont=dict(color="white"))

    # ════════════════════════════════════════════════════════════════════════
    # Global layout
    # ════════════════════════════════════════════════════════════════════════
    fig.update_layout(
        height=1200,
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        font=dict(color="white", family="Inter, Arial, sans-serif"),
        title=dict(
            text=(
                "<b>Escape Room — Leadership & Social Interaction Analysis</b><br>"
                "<sup>Bass & Avolio (1994) · Bales IPA · Bavelas centrality · Mast talk-time · Hollander idiosyncrasy credits</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=16, color="white"),
        ),
        legend=dict(
            bgcolor="rgba(20,20,40,0.8)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=11),
        ),
        barmode="group",
        annotations=[
            # Key / legend box for graph
            dict(
                text=(
                    "<b>KEY</b><br>"
                    "<span style='color:#e94560'>●</span> A  "
                    "<span style='color:#0f9b8e'>●</span> B  "
                    "<span style='color:#f5a623'>●</span> C  "
                    "<span style='color:#7b5ea7'>●</span> D<br>"
                    "Node size = emergent leadership score<br>"
                    "Edge width = utterance count<br>"
                    "Edge curve = communication direction<br>"
                    "<span style='color:#00d4aa'>●</span> positive  "
                    "<span style='color:#e94560'>●</span> negative  "
                    "<span style='color:#ccc'>●</span> neutral turn<br>"
                    "<b>★</b> = leadership moment"
                ),
                x=0.35, y=0.96,
                xref="paper", yref="paper",
                align="left",
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(20,20,40,0.75)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                borderpad=6,
            ),
            # Moment legend
            dict(
                text=(
                    "<b>LEADERSHIP MOMENTS</b><br>" +
                    "  ".join(
                        f"<span style='color:{c}'>★</span> {k}"
                        for k, c in MOMENT_COLORS.items()
                    )
                ),
                x=0.005, y=0.36,
                xref="paper", yref="paper",
                align="left",
                showarrow=False,
                font=dict(size=9, color="white"),
                bgcolor="rgba(20,20,40,0.75)",
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                borderpad=5,
            ),
            # Style legend
            dict(
                text=(
                    "<b>LEADERSHIP STYLES</b> (Bass & Avolio 1994 + Yukl 2012)<br>"
                    "Radar: Transformational · Transactional · IPA Task · Socio+ · Insight · Talk share<br>"
                    "Score = 40% network centrality + 20% talk share + 15% transformational + 15% transactional + 10% insight"
                ),
                x=0.63, y=0.36,
                xref="paper", yref="paper",
                align="left",
                showarrow=False,
                font=dict(size=9, color="#aaaaaa"),
                bgcolor="rgba(20,20,40,0.75)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
                borderpad=5,
            ),
        ],
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        if ann.text in (
            "Social Graph — Who Responds to Whom",
            "Leadership Radar",
            "Discrete Interaction Timeline",
            "Leadership Moment Map",
            "Composite Emergent Score",
        ):
            ann.font.color = "rgba(200,200,200,0.8)"
            ann.font.size = 11

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[graph_enhanced] wrote {out_path}")
    return out_path
