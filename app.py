"""
Psychological Dialogue Analysis Dashboard
==========================================
Three input modes:
  1. Upload Excel / CSV  — manual coded dialogue
  2. Drop Video files    — auto-transcribe with WhisperX (GPU)
  3. Load saved JSON     — reload a previous run

Launch:  streamlit run app.py   (or double-click run.bat)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))
from src.excel_parser import load_dialogue
from src.social_graph import build_graph, compute_metrics, erdos_distances, export_html
from src.psych_analysis import annotate_turns, speaker_profiles, emotional_contagion_matrix
from src.leadership_assessment import full_assessment
from src.graph_enhanced import export_enhanced_html

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dialogue Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .metric-card { background: #1e1e2e; border-radius: 8px; padding: 1rem; margin: 0.25rem; }
    section[data-testid="stSidebar"] { background: #12121e; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Dialogue Analysis")
    st.divider()

    input_mode = st.radio(
        "Input mode",
        ["Excel / CSV", "Video files (auto-transcribe)", "Load saved JSON"],
        index=0,
    )

    st.divider()
    session_id  = st.text_input("Session ID", value="session_01")
    use_emotion = st.checkbox("Emotion model (GPU)", value=True,
                              help="RTX 5060. First run downloads ~300MB.")

    # ── Mode: Excel / CSV ──────────────────────────────────────────────────
    if input_mode == "Excel / CSV":
        uploaded = st.file_uploader(
            "Dialogue sheet (.xlsx / .xls / .csv)",
            type=["xlsx", "xls", "csv"],
            help="Required: speaker, utterance. Optional: turn, timestamp, code",
        )
        sheet_name = st.text_input("Sheet name (blank = first sheet)", value="")
        run_btn    = st.button("Analyse", type="primary", use_container_width=True)

    # ── Mode: Video ────────────────────────────────────────────────────────
    elif input_mode == "Video files (auto-transcribe)":
        uploaded     = None
        video_files  = st.file_uploader(
            "MP4 video files (one or more)",
            type=["mp4", "mov", "mkv"],
            accept_multiple_files=True,
        )
        whisper_model = st.selectbox(
            "Transcription model",
            ["large-v3-turbo", "large-v3", "medium", "small", "base"],
            index=0,
            help="large-v3-turbo: best speed/quality on RTX 5060 (3 GB VRAM)",
        )
        cam_map_raw = st.text_area(
            "Headcam player map (filename=Player, one per line)",
            value="cam_A.mp4=A\ncam_B.mp4=B\ncam_C.mp4=C\ncam_D.mp4=D",
            help="Leave entries for security cameras blank — they use full face+diarization",
        )
        player_order = st.text_input(
            "Player order for audio diarization (security cams)",
            value="A,B,C,D",
        )
        hf_token = st.text_input(
            "HuggingFace token (for speaker diarization)",
            value=os.environ.get("HF_TOKEN", ""),
            type="password",
            help="Free at huggingface.co — needed for multi-speaker diarization",
        )
        run_face = st.checkbox("Face detection (InsightFace + MediaPipe)", value=True)
        run_btn  = st.button("Transcribe + Analyse", type="primary",
                              use_container_width=True)

    # ── Mode: Load JSON ────────────────────────────────────────────────────
    else:
        uploaded    = None
        json_file   = st.file_uploader("Annotated turns JSON", type=["json"])
        run_btn     = st.button("Load + Analyse", type="primary",
                                use_container_width=True)

    st.divider()
    st.caption("RTX 5060 · 8 GB VRAM · CUDA 12.8")

# ── Main ───────────────────────────────────────────────────────────────────────

# Determine if we have anything to process
has_input = (
    (input_mode == "Excel / CSV" and uploaded) or
    (input_mode == "Video files (auto-transcribe)" and video_files) or
    (input_mode == "Load saved JSON" and json_file)
)

if not has_input:
    st.title("Psychological Dialogue Analysis")
    st.markdown("""
    ### Three ways to get started

    | Mode | How | When |
    |---|---|---|
    | **Excel / CSV** | Upload a coded dialogue sheet | You have manual transcripts |
    | **Video files** | Drop MP4s — WhisperX transcribes on your RTX 5060 | Raw footage, no manual work |
    | **Load JSON** | Reload a previous run | Resume or re-visualise |

    ### What you get
    | Output | Description |
    |---|---|
    | **Enhanced graph** | Cube-layout social network with discrete interaction events |
    | **Leadership scores** | Bass & Avolio · Bales IPA · Bavelas centrality · Mast talk-time |
    | **Leadership moments** | Directive / Insight / Consensus / Coordination / Breakthrough |
    | **Sentiment timeline** | Turn-by-turn positivity per speaker |
    | **Emotion distribution** | GPU-classified joy / anger / sadness / fear / surprise |
    | **Emotional contagion** | Does A's mood shift B's next response? |
    | **Erdos distances** | Social distance matrix |
    | **Raw data** | Downloadable annotated CSV + JSON |
    """)
    st.stop()

if not run_btn:
    st.info("Upload complete. Click the Analyse button in the sidebar to run.")
    st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────────

turns = None

# --- Excel / CSV ---
if input_mode == "Excel / CSV":
    with st.spinner("Parsing file..."):
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            sheet_arg = sheet_name.strip() if sheet_name.strip() else 0
            turns = load_dialogue(tmp_path, sheet=sheet_arg,
                                  session_id=session_id.strip() or None)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.stop()
    st.toast(f"Loaded {len(turns)} turns from {len({t['speaker'] for t in turns})} speakers")

# --- Load JSON ---
elif input_mode == "Load saved JSON":
    with st.spinner("Loading JSON..."):
        try:
            turns = json.load(json_file)
        except Exception as e:
            st.error(f"Could not load JSON: {e}")
            st.stop()
    st.toast(f"Loaded {len(turns)} turns")

# --- Video pipeline ---
elif input_mode == "Video files (auto-transcribe)":
    try:
        from src.video_pipeline import process_session
    except ImportError:
        st.error("Video stack not installed. Run install_video_stack.bat first.")
        st.stop()

    # Save uploaded videos to temp dir
    tmp_dir = Path(tempfile.mkdtemp())
    video_paths = []
    for vf in video_files:
        dest = tmp_dir / vf.name
        dest.write_bytes(vf.read())
        video_paths.append(dest)

    # Parse headcam map
    camera_to_player = {}
    for line in cam_map_raw.strip().splitlines():
        line = line.strip()
        if "=" in line:
            fname, player = line.split("=", 1)
            camera_to_player[fname.strip()] = player.strip()

    player_labels = [p.strip() for p in player_order.split(",") if p.strip()]
    out_dir = Path("output") / (session_id or "video_session")

    progress_bar = st.progress(0, text="Initialising video pipeline...")

    def _progress_cb(step, pct):
        progress_bar.progress(min(int(pct * 100), 100), text=step)

    with st.spinner("Running video pipeline on GPU (this takes several minutes)..."):
        try:
            turns = process_session(
                video_paths=video_paths,
                out_dir=out_dir,
                session_id=session_id,
                camera_to_player=camera_to_player or None,
                player_labels=player_labels,
                whisper_model=whisper_model,
                hf_token=hf_token,
                run_face_id=run_face,
                progress_callback=_progress_cb,
            )
        except Exception as e:
            st.error(f"Video pipeline error: {e}")
            st.stop()

    progress_bar.progress(100, text="Transcription complete")
    st.toast(f"Transcribed {len(turns)} turns from {len(video_files)} video(s)")

if turns is None or len(turns) == 0:
    st.error("No turns loaded.")
    st.stop()

with st.spinner("Running NLP + emotion analysis on GPU..."):
    turns = annotate_turns(turns, use_emotion_model=use_emotion)

with st.spinner("Building social graph + leadership assessment..."):
    G        = build_graph(turns)
    metrics  = compute_metrics(G)
    profiles = speaker_profiles(turns)
    contagion = emotional_contagion_matrix(turns)
    dist_df  = erdos_distances(G)
    ld_scores, moments, composites, styles = full_assessment(turns, metrics, profiles)

    # Enhanced graph HTML
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as gh:
        enhanced_html_path = gh.name
    export_enhanced_html(
        G=G, turns=turns, metrics=metrics,
        ld_scores=ld_scores, moments=moments,
        composites=composites, styles=styles, profiles=profiles,
        out_path=enhanced_html_path,
    )
    enhanced_html = Path(enhanced_html_path).read_text(encoding="utf-8")

# ── Tabs ───────────────────────────────────────────────────────────────────────
speakers = sorted(profiles.keys())
tab_graph, tab_leaders, tab_sentiment, tab_emotion, tab_contagion, tab_erdos, tab_data = st.tabs([
    "Social Graph + Leadership",
    "Leadership Deep-Dive",
    "Sentiment",
    "Emotions",
    "Contagion",
    "Erdos Distances",
    "Raw Data",
])

# ── Social Graph tab ──────────────────────────────────────────────────────────
with tab_graph:
    st.subheader("Interactive Social Graph")
    st.caption(
        "Cube layout · node size = composite emergent leadership score · "
        "dots on edges = individual turns colored by sentiment · "
        "stars = leadership moments (hover for detail)"
    )
    components.html(enhanced_html, height=1250, scrolling=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Speakers", len(G.nodes))
    c2.metric("Directed edges", len(G.edges))
    c3.metric("Leadership moments", len(moments))
    c4.metric("Total turns", len(turns))

# ── Leadership deep-dive tab ──────────────────────────────────────────────────
with tab_leaders:
    st.subheader("Leadership Assessment")
    st.caption(
        "Bass & Avolio (1994) Full Range Leadership · Bales (1950) IPA · "
        "Bavelas (1950) network centrality · Mast (2002) talk-time · "
        "Hollander (1958) idiosyncrasy credits"
    )

    # Composite scores
    ranked = sorted(composites.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(len(ranked))
    for col, (spk, score) in zip(cols, ranked):
        col.metric(
            f"Player {spk}",
            f"{score:.2f}",
            delta=styles.get(spk, ""),
            delta_color="off",
        )

    # ── Full Range Leadership radar ────────────────────────────────────────
    st.subheader("Full Range Leadership Dimensions (Bass & Avolio 1994)")
    PLAYER_COLORS = {"A": "#e94560", "B": "#0f9b8e", "C": "#f5a623", "D": "#7b5ea7"}
    dimensions = ["transformational", "transactional", "laissez_faire",
                  "ipa_task", "ipa_socio_pos", "insight_rate"]
    dim_labels  = ["Transformational", "Transactional", "Laissez-faire",
                   "IPA Task", "IPA Socio+", "Insight Rate"]

    radar_fig = go.Figure()
    for spk in speakers:
        sc = ld_scores.get(spk, {})
        # Normalize each dimension by its max across speakers for radar readability
        vals = [sc.get(d, 0) for d in dimensions]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dim_labels + [dim_labels[0]],
            name=f"Player {spk}",
            fill="toself",
            opacity=0.6,
            line_color=PLAYER_COLORS.get(spk, "#aaa"),
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor="#333", linecolor="#333",
                                   tickfont=dict(color="#aaa")),
                   angularaxis=dict(gridcolor="#333", linecolor="#333")),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
        legend=dict(bgcolor="#0e1117"),
        height=420,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # ── Leadership dimensions heatmap ──────────────────────────────────────
    dim_rows = []
    for spk in speakers:
        sc = ld_scores.get(spk, {})
        dim_rows.append({
            "Speaker": f"Player {spk}",
            "Transformational": round(sc.get("transformational", 0), 3),
            "Transactional":    round(sc.get("transactional", 0), 3),
            "Laissez-faire":    round(sc.get("laissez_faire", 0), 3),
            "IPA Task":         round(sc.get("ipa_task", 0), 3),
            "IPA Socio+":       round(sc.get("ipa_socio_pos", 0), 3),
            "IPA Socio-":       round(sc.get("ipa_socio_neg", 0), 3),
            "Insight Rate":     round(sc.get("insight_rate", 0), 3),
            "Talk Share":       round(sc.get("talk_share", 0), 3),
            "Style":            styles.get(spk, ""),
            "Composite":        round(composites.get(spk, 0), 3),
        })
    dim_df = pd.DataFrame(dim_rows).set_index("Speaker")
    st.dataframe(dim_df, use_container_width=True)

    # ── Talk share bar ─────────────────────────────────────────────────────
    talk_data = pd.DataFrame([
        {"Speaker": f"Player {s}", "Talk Share %": round(ld_scores.get(s, {}).get("talk_share", 0) * 100, 1),
         "Color": PLAYER_COLORS.get(s, "#aaa")}
        for s in speakers
    ])
    ts_fig = px.bar(
        talk_data, x="Speaker", y="Talk Share %",
        color="Speaker",
        color_discrete_map={f"Player {s}": PLAYER_COLORS.get(s, "#aaa") for s in speakers},
        title="Talk Share (Mast 2002: speaking time ∝ dominance)",
    )
    ts_fig.update_layout(showlegend=False, plot_bgcolor="#0e1117",
                         paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(ts_fig, use_container_width=True)

    # ── Leadership moments ─────────────────────────────────────────────────
    st.subheader(f"Leadership Moments ({len(moments)} total)")
    if moments:
        mom_df = pd.DataFrame([{
            "Turn":     m.get("turn", ""),
            "Speaker":  f"Player {m.get('speaker', '')}",
            "Type":     m.get("type", ""),
            "Utterance": m.get("utterance", "")[:100],
        } for m in moments])
        st.dataframe(mom_df, use_container_width=True, height=350)

        # Frequency breakdown
        mom_counts = mom_df.groupby(["Speaker", "Type"]).size().reset_index(name="Count")
        mf_fig = px.bar(
            mom_counts, x="Type", y="Count", color="Speaker",
            barmode="group",
            color_discrete_map={f"Player {s}": PLAYER_COLORS.get(s, "#aaa") for s in speakers},
            title="Leadership Moment Frequency by Type and Speaker",
        )
        mf_fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white")
        st.plotly_chart(mf_fig, use_container_width=True)
    else:
        st.info("No leadership moments detected.")

    st.divider()
    st.subheader("Network Centrality (Bavelas / Freeman)")

    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Speaker"})
    metrics_df = metrics_df.sort_values("eigenvector_centrality", ascending=False)

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(
            metrics_df, x="Speaker", y="eigenvector_centrality",
            color="eigenvector_centrality",
            color_continuous_scale="Reds",
            title="Eigenvector Centrality (Leadership Emergence)",
            labels={"eigenvector_centrality": "Score"},
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            metrics_df, x="Speaker", y="pagerank",
            color="pagerank",
            color_continuous_scale="Blues",
            title="PageRank (Influence Propagation)",
            labels={"pagerank": "Score"},
        )
        fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        metrics_df,
        x="dominance_ratio", y="betweenness_centrality",
        size="turn_count", color="eigenvector_centrality",
        text="Speaker",
        color_continuous_scale="Plasma",
        title="Dominance vs. Brokerage (bubble size = turn count)",
        labels={
            "dominance_ratio": "Dominance Ratio (in/out degree)",
            "betweenness_centrality": "Betweenness (Gatekeeper score)",
        },
    )
    fig3.update_traces(textposition="top center")
    fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(metrics_df.set_index("Speaker").round(4), use_container_width=True)

# ── Sentiment tab ─────────────────────────────────────────────────────────────
with tab_sentiment:
    st.subheader("Sentiment Timeline")
    sent_df = pd.DataFrame([
        {"Turn": t["turn"], "Speaker": t["speaker"],
         "Sentiment": t.get("sentiment", 0),
         "Utterance": t.get("utterance", "")[:80]}
        for t in turns
    ])

    fig = px.line(
        sent_df, x="Turn", y="Sentiment", color="Speaker",
        hover_data=["Utterance"],
        title="Turn-by-turn Sentiment per Speaker",
    )
    fig.add_hline(y=0.05, line_dash="dot", line_color="green", opacity=0.4)
    fig.add_hline(y=-0.05, line_dash="dot", line_color="red", opacity=0.4)
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean Sentiment per Speaker")
    mean_sent = metrics_df[["Speaker"]].copy()
    mean_sent["Mean Sentiment"] = [profiles[s]["mean_sentiment"] for s in mean_sent["Speaker"]]
    mean_sent["Std"] = [profiles[s]["sentiment_std"] for s in mean_sent["Speaker"]]
    fig2 = px.bar(mean_sent, x="Speaker", y="Mean Sentiment", error_y="Std",
                  color="Mean Sentiment", color_continuous_scale="RdYlGn",
                  title="Mean Sentiment (± 1 SD)")
    fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font_color="white", coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Verbal Dominance Breakdown")
    dom_rows = []
    for spk in speakers:
        p = profiles[spk]
        dom_rows.append({
            "Speaker": spk,
            "Assertive %": p["assertive_pct"] * 100,
            "Directive %": p["directive_pct"] * 100,
            "Hedging %":   p["hedging_pct"] * 100,
            "Agreeable %": p["agreeable_pct"] * 100,
            "Interrupting %": p["interrupting_pct"] * 100,
        })
    dom_df = pd.DataFrame(dom_rows).set_index("Speaker")
    fig3 = px.bar(dom_df.reset_index(), x="Speaker",
                  y=["Assertive %", "Directive %", "Hedging %", "Agreeable %", "Interrupting %"],
                  barmode="group",
                  title="Linguistic Dominance Markers (%)",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig3, use_container_width=True)

# ── Emotion tab ───────────────────────────────────────────────────────────────
with tab_emotion:
    st.subheader("Emotion Distribution per Speaker")
    if not any(t.get("emotion") for t in turns):
        st.info("Emotion model was not run. Re-run with 'Run emotion model (GPU)' checked.")
    else:
        rows = []
        for spk in speakers:
            for em, pct in profiles[spk].get("emotion_distribution", {}).items():
                rows.append({"Speaker": spk, "Emotion": em, "Proportion": pct})
        em_df = pd.DataFrame(rows)

        fig = px.bar(em_df, x="Speaker", y="Proportion", color="Emotion",
                     barmode="stack",
                     title="Emotion Profile per Speaker (stacked)",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Turn-level Emotion Timeline")
        em_time = pd.DataFrame([
            {"Turn": t["turn"], "Speaker": t["speaker"], "Emotion": t.get("emotion", "none")}
            for t in turns if t.get("emotion")
        ])
        fig2 = px.scatter(em_time, x="Turn", y="Speaker", color="Emotion", symbol="Emotion",
                          title="Emotion per Turn",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

# ── Contagion tab ────────────────────────────────────────────────────────────
with tab_contagion:
    st.subheader("Emotional Contagion Matrix")
    st.caption("Cell value: mean change in B's sentiment in the turn immediately after an emotionally charged turn by A. Positive = A's emotional charge makes B more positive.")

    if not contagion:
        st.info("Not enough emotionally charged turns to compute contagion (need turns with |sentiment| ≥ 0.05).")
    else:
        cont_rows = []
        for a, targets in contagion.items():
            for b, delta in targets.items():
                cont_rows.append({"From (A)": a, "To (B)": b, "Mean Δ Sentiment": delta})
        cont_df = pd.DataFrame(cont_rows)
        pivot = cont_df.pivot(index="To (B)", columns="From (A)", values="Mean Δ Sentiment")

        fig = px.imshow(
            pivot, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            title="Emotional Contagion: row = who is affected, col = who causes it",
        )
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

# ── Erdős Distances tab ───────────────────────────────────────────────────────
with tab_erdos:
    st.subheader("Erdős Distances (Social Distance Matrix)")
    st.caption("Number of interaction hops between any two speakers. 1 = they responded directly to each other. ∞ = no path.")

    display_dist = dist_df.copy().astype(object)
    for col in display_dist.columns:
        display_dist[col] = display_dist[col].apply(
            lambda x: "∞" if x == float("inf") else int(x) if x == 0 else x
        )
    st.dataframe(display_dist, use_container_width=True)

    numeric_dist = dist_df.replace(float("inf"), None)
    fig = px.imshow(
        numeric_dist, text_auto=True, aspect="auto",
        color_continuous_scale="Viridis_r",
        title="Social Distance Heatmap (darker = closer)",
    )
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

# ── Raw Data tab ──────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Annotated Dialogue Data")
    turns_df = pd.DataFrame(turns)
    st.dataframe(turns_df, use_container_width=True)

    csv = turns_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download annotated CSV",
        data=csv,
        file_name=f"{session_id or 'session'}_annotated.csv",
        mime="text/csv",
    )

    st.subheader("Speaker Profiles (JSON)")
    st.json(profiles)

    prof_df = pd.DataFrame(profiles).T
    prof_csv = prof_df.drop(columns=["emotion_distribution"], errors="ignore").to_csv().encode("utf-8")
    st.download_button(
        "⬇ Download speaker profiles CSV",
        data=prof_csv,
        file_name=f"{session_id or 'session'}_profiles.csv",
        mime="text/csv",
    )
