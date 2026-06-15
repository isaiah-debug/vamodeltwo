"""Manual smoke test for the focused transcript CSV pipeline.

Run:
    python scripts/test_pipeline.py
"""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.addressee_inference import infer_addressees
from src.export_utterances import export_utterances_csv


def main() -> int:
    turns = [
        {
            "turn": 1,
            "session": "smoke",
            "speaker": "A",
            "start_s": 1.0,
            "end_s": 2.0,
            "utterance": "Everyone look at this clue.",
        },
        {
            "turn": 2,
            "session": "smoke",
            "speaker": "B",
            "start_s": 3.0,
            "end_s": 4.0,
            "utterance": "Jordan, can you try the red lock?",
        },
    ]
    enriched = infer_addressees(
        turns,
        players=["A", "B"],
        player_aliases={"A": ["Jordan"], "B": ["Elis"]},
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = export_utterances_csv(enriched, Path(tmp) / "utterances.csv")
        print(out.read_text(encoding="utf-8"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
