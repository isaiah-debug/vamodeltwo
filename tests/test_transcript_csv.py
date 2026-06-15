import csv
import json
import argparse

import pytest

from scripts.transcribe_to_csv import main, parse_assignments
from src.addressee_inference import infer_addressees
from src.export_utterances import UTTERANCE_COLUMNS, export_utterances_csv
from src.multi_csv_loader import load_experiment, parse_receivers, parse_timestamp


def test_infer_addressees_uses_auditable_methods():
    turns = [
        {"turn": 1, "speaker": "A", "utterance": "Everyone look at the wall."},
        {"turn": 2, "speaker": "B", "utterance": "Jordan, can you read this?"},
        {"turn": 3, "speaker": "A", "utterance": "Can you pass me the key?"},
        {"turn": 4, "speaker": "C", "utterance": "The red lock is open."},
    ]

    enriched = infer_addressees(
        turns,
        players=["A", "B", "C"],
        player_aliases={"A": ["Jordan"], "B": ["Elis"], "C": ["Anna"]},
    )

    assert enriched[0]["addressee"] == "All"
    assert enriched[0]["addressee_method"] == "broadcast_keyword"
    assert enriched[1]["addressee"] == "A"
    assert enriched[1]["addressee_method"] == "name_mention"
    assert enriched[2]["addressee"] == "B"
    assert enriched[2]["addressee_method"] == "pronoun_context"
    assert enriched[3]["addressee"] == "A"
    assert enriched[3]["addressee_method"] == "sequential_context"


def test_infer_addressees_preserves_coded_receivers():
    turns = [
        {
            "turn": 1,
            "speaker": "A",
            "receiver_raw": "B and C",
            "utterance": "Try these two locks.",
        },
        {
            "turn": 2,
            "speaker": "B",
            "receiver_raw": "All",
            "utterance": "Everyone come here.",
        },
    ]

    enriched = infer_addressees(turns, players=["A", "B", "C"])

    assert enriched[0]["addressee"] == "B;C"
    assert enriched[0]["addressee_method"] == "coded"
    assert enriched[1]["addressee"] == "All"
    assert enriched[1]["addressee_method"] == "coded"


def test_export_utterances_csv_column_order_and_timestamp(tmp_path):
    turns = [
        {
            "session": "session_01",
            "turn": 1,
            "start_s": 61.25,
            "end_s": 63.0,
            "speaker": "A",
            "addressee": "B",
            "utterance": "Can you hold this?",
            "addressee_method": "pronoun_context",
        }
    ]

    out = export_utterances_csv(turns, tmp_path / "utterances.csv")

    with out.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(UTTERANCE_COLUMNS)
    assert rows[0]["timestamp"] == "01:01"
    assert rows[0]["start_s"] == "61.25"
    assert rows[0]["end_s"] == "63"


def test_parse_timestamp_and_receivers_cover_coding_formats():
    assert parse_timestamp("0:35") == 35.0
    assert parse_timestamp("10:51") == 651.0
    assert parse_timestamp("0056") == 56.0
    assert parse_timestamp("24:17:00") == 1457.0

    assert parse_receivers("All", {"A", "B", "C"}) == ["ALL"]
    assert parse_receivers("B and C", {"A", "B", "C"}) == ["B", "C"]
    assert parse_receivers("self", {"A", "B", "C"}) == []
    assert parse_receivers("B?", {"A", "B", "C"}) == ["B"]


def test_load_experiment_and_export_round_trip_coded_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "player_A_Jordan.csv").write_text(
        "speaker,receiver,start,end,transcript,notes_challenges\n"
        "A,All,0:01,0:02,Which one first?,\n"
        ",B,0:07,0:10,You want to split up?,\n",
        encoding="utf-8",
    )
    (data_dir / "player_B_Elis.csv").write_text(
        "speaker,receiver,start,end,transcript,notes_challenges\n"
        "B,A,0:03,0:04,Try the red one.,\n",
        encoding="utf-8",
    )

    turns, _edges = load_experiment(data_dir, session_id="escape_room_01")
    enriched = infer_addressees(turns, players=["A", "B"])
    out = export_utterances_csv(enriched, tmp_path / "utterances.csv")

    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["speaker"] for row in rows] == ["A", "B", "A"]
    assert [row["addressee"] for row in rows] == ["All", "A", "B"]
    assert all(row["addressee_method"] == "coded" for row in rows)


def test_transcribe_to_csv_from_json_does_not_need_video_stack(tmp_path):
    turns_path = tmp_path / "video_turns.json"
    turns_path.write_text(
        json.dumps(
            [
                {
                    "turn": 1,
                    "session": "session_01",
                    "speaker": "A",
                    "start_s": 1,
                    "end_s": 2,
                    "utterance": "Elis, look here.",
                }
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "utterances.csv"

    exit_code = main(
        [
            "--from-json",
            str(turns_path),
            "--players",
            "A",
            "B",
            "--player-name",
            "B=Elis",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["addressee"] == "B"
    assert rows[0]["addressee_method"] == "name_mention"


def test_parse_assignments_rejects_invalid_pairs():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_assignments(["A"])
