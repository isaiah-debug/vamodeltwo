import argparse
import csv
import json

import pytest

from scripts.transcribe_to_csv import main, parse_assignments, parse_offsets, parse_track_map
from src.addressee_inference import infer_addressees
from src.export_utterances import UTTERANCE_COLUMNS, export_utterances_csv
from src.video_pipeline import process_multitrack_session, seconds_to_timestamp


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


def test_export_utterances_csv_includes_source_and_track_metadata(tmp_path):
    turns = [
        {
            "session": "session_01",
            "turn": 1,
            "source_file": "session_part1.mp4",
            "audio_track": 2,
            "start_s": 61.25,
            "end_s": 63.0,
            "local_start_s": 1.25,
            "local_end_s": 3.0,
            "speaker": "C",
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
    assert rows[0]["source_file"] == "session_part1.mp4"
    assert rows[0]["audio_track"] == "2"
    assert rows[0]["timestamp"] == "01:01"
    assert rows[0]["start_s"] == "61.25"
    assert rows[0]["end_s"] == "63"


def test_parse_track_map_and_offsets():
    assert parse_track_map(None, ["A", "B", "C", "D"]) == {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
    }
    assert parse_track_map(["0=A", "1=B", "2=C", "3=D"], []) == {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
    }
    assert parse_offsets(["part2.mp4=1800", "part3.mp4=3600"]) == {
        "part2.mp4": 1800.0,
        "part3.mp4": 3600.0,
    }


def test_transcribe_to_csv_from_json_does_not_need_video_stack(tmp_path):
    turns_path = tmp_path / "turns.json"
    turns_path.write_text(
        json.dumps(
            [
                {
                    "turn": 1,
                    "session": "session_01",
                    "source_file": "session_part1.mp4",
                    "audio_track": 0,
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


def test_media_cli_validates_expected_file_and_track_counts(tmp_path):
    media = tmp_path / "one.mp4"
    media.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Expected 7 MP4 files"):
        main(["--media", str(media), "--track-map", "0=A", "1=B", "2=C", "3=D"])

    with pytest.raises(ValueError, match="Expected 4 speaker audio tracks"):
        main(["--media", str(media), "--expected-files", "1", "--track-map", "0=A"])


def test_process_multitrack_session_merges_offsets_and_tracks(tmp_path, monkeypatch):
    media1 = tmp_path / "part1.mp4"
    media2 = tmp_path / "part2.mp4"
    media1.write_bytes(b"one")
    media2.write_bytes(b"two")

    def fake_extract(media_path, out_dir, track_index, sample_rate=16000):
        wav = tmp_path / f"{media_path.stem}_track{track_index}.wav"
        wav.write_bytes(b"wav")
        return wav

    def fake_transcribe(
        wav_path,
        out_dir,
        speaker,
        source_file,
        track_index,
        session_id,
        offset_s=0.0,
        **_kwargs,
    ):
        local_start = float(track_index)
        start_s = offset_s + local_start
        return [
            {
                "session": session_id,
                "source_file": source_file,
                "audio_track": track_index,
                "speaker": speaker,
                "utterance": f"{speaker} says hello",
                "timestamp": seconds_to_timestamp(start_s),
                "start_s": start_s,
                "end_s": start_s + 1,
                "local_start_s": local_start,
                "local_end_s": local_start + 1,
            }
        ]

    monkeypatch.setattr("src.video_pipeline.extract_audio_track", fake_extract)
    monkeypatch.setattr("src.video_pipeline.transcribe_isolated_wav", fake_transcribe)

    turns = process_multitrack_session(
        media_paths=[media1, media2],
        track_map={0: "A", 1: "B"},
        out_dir=tmp_path / "out",
        session_id="session_01",
        file_offsets={"part2.mp4": 1800},
    )

    assert [turn["turn"] for turn in turns] == [1, 2, 3, 4]
    assert [turn["speaker"] for turn in turns] == ["A", "B", "A", "B"]
    assert turns[2]["source_file"] == "part2.mp4"
    assert turns[2]["start_s"] == 1800.0


def test_parse_assignments_rejects_invalid_pairs():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_assignments(["A"])
