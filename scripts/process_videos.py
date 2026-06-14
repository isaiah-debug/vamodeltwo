"""
One-command video session processor.

Usage (headcam shortcut — fastest):
    python scripts/process_videos.py --config configs/project_template.yaml

Usage (explicit video list):
    python scripts/process_videos.py \\
        --videos data/videos/cam_A.mp4 data/videos/cam_B.mp4 \\
        --camera-map cam_A.mp4=A cam_B.mp4=B \\
        --session my_session_01

Usage (security cameras, full face pipeline):
    python scripts/process_videos.py \\
        --videos data/videos/sec_1.mp4 \\
        --players A B C D \\
        --session sec_01

After this script finishes, run the analysis dashboard:
    streamlit run app.py
Then upload the generated JSON from output/<session>/video_turns.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",     help="Path to project YAML config")
    ap.add_argument("--videos",     nargs="+", help="Video file paths")
    ap.add_argument("--camera-map", nargs="+", metavar="FILE=PLAYER",
                    help="headcam assignments, e.g. cam_A.mp4=A")
    ap.add_argument("--players",    nargs="+", default=["A","B","C","D"],
                    help="Player labels for audio diarization order")
    ap.add_argument("--session",    default="session_01")
    ap.add_argument("--model",      default="large-v3-turbo",
                    help="WhisperX model (tiny/base/small/medium/large-v3/large-v3-turbo)")
    ap.add_argument("--no-face",    action="store_true",
                    help="Skip face detection (audio diarization only)")
    ap.add_argument("--hf-token",   default="",
                    help="HuggingFace token for pyannote diarization")
    args = ap.parse_args()

    # ── Load config if provided ────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        session_id  = cfg["project"]["session_id"]
        player_labels = [p["label"] for p in cfg["players"]]
        data_dir    = Path(cfg["videos"]["data_dir"])
        out_dir     = Path(cfg["output"]["dir"]) / session_id
        model       = cfg["transcription"]["whisper_model"]
        run_face    = cfg["face_id"]["enabled"]
        sample_n    = cfg["face_id"]["sample_every_n_frames"]
        min_spk     = cfg["transcription"]["min_speakers"]
        max_spk     = cfg["transcription"]["max_speakers"]
        language    = cfg["transcription"]["language"]
        hf_token    = args.hf_token or os.environ.get("HF_TOKEN", "")

        # Headcam map
        camera_to_player = {}
        for hc in cfg["videos"].get("headcams", []):
            camera_to_player[hc["file"]] = hc["player"]

        # Build video list: headcams first, then security cameras
        video_paths = []
        for hc in cfg["videos"].get("headcams", []):
            p = data_dir / hc["file"]
            if p.exists():
                video_paths.append(p)
            else:
                print(f"  WARNING: headcam not found: {p}")
        for sc in cfg["videos"].get("security_cameras", []):
            p = data_dir / sc["file"]
            if p.exists():
                video_paths.append(p)
            else:
                print(f"  WARNING: security cam not found: {p}")

    else:
        # ── CLI mode ───────────────────────────────────────────────────────
        if not args.videos:
            ap.error("Provide --config or --videos")

        session_id    = args.session
        player_labels = args.players
        video_paths   = [Path(v) for v in args.videos]
        out_dir       = Path("output") / session_id
        model         = args.model
        run_face      = not args.no_face
        sample_n      = 15
        min_spk       = 2
        max_spk       = len(player_labels)
        language      = "en"
        hf_token      = args.hf_token or os.environ.get("HF_TOKEN", "")

        camera_to_player = {}
        if args.camera_map:
            for item in args.camera_map:
                fname, player = item.split("=")
                camera_to_player[fname] = player

    if not video_paths:
        print("No video files found. Check your config paths.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"VIDEO SESSION PROCESSOR")
    print(f"{'='*60}")
    print(f"Session  : {session_id}")
    print(f"Videos   : {[v.name for v in video_paths]}")
    print(f"Model    : {model}  (GPU: RTX 5060)")
    print(f"Face ID  : {'yes' if run_face else 'no (audio-only)'}")
    print(f"Players  : {player_labels}")
    print(f"Output   : {out_dir}")
    print()

    from src.video_pipeline import process_session
    turns = process_session(
        video_paths=video_paths,
        out_dir=out_dir,
        session_id=session_id,
        camera_to_player=camera_to_player or None,
        player_labels=player_labels,
        whisper_model=model,
        language=language,
        min_speakers=min_spk,
        max_speakers=max_spk,
        hf_token=hf_token,
        run_face_id=run_face,
        sample_video_every_n=sample_n,
    )

    # Run full analysis pipeline
    print(f"\n{'='*60}")
    print("RUNNING ANALYSIS PIPELINE")
    print(f"{'='*60}")

    from src.psych_analysis import annotate_turns, speaker_profiles
    from src.multi_csv_loader import build_explicit_graph
    from src.social_graph import compute_metrics, erdos_distances
    from src.leadership_assessment import full_assessment
    from src.graph_enhanced import export_enhanced_html

    print(f"[1/4] NLP annotation ({len(turns)} turns)...")
    turns = annotate_turns(turns, use_emotion_model=True, batch_size=32)

    print("[2/4] Social graph...")
    # For video-only (no receiver info), use sequential adjacency graph
    from src.social_graph import build_graph, compute_metrics
    G        = build_graph(turns)
    metrics  = compute_metrics(G)
    profiles = speaker_profiles(turns)

    print("[3/4] Leadership assessment...")
    ld_scores, moments, composites, styles = full_assessment(turns, metrics, profiles)
    for spk in sorted(styles):
        print(f"  Player {spk}: {styles[spk]}  (score={composites[spk]:.3f})")

    print("[4/4] Generating visualization...")
    export_enhanced_html(
        G=G, turns=turns, metrics=metrics,
        ld_scores=ld_scores, moments=moments,
        composites=composites, styles=styles, profiles=profiles,
        out_path=out_dir / "enhanced_analysis.html",
    )

    # Save turns for dashboard
    with open(out_dir / "video_turns_annotated.json", "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2, ensure_ascii=False)

    print(f"\nDONE. Open:")
    print(f"  {(out_dir / 'enhanced_analysis.html').resolve()}")
    print(f"\nOr run the dashboard:")
    print(f"  streamlit run app.py")


if __name__ == "__main__":
    main()
