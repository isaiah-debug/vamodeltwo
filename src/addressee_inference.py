"""Infer addressees for participant utterances.

The video pipeline can identify when each participant speaks, but "who they
were speaking to" is not directly observable from audio alone. This module
keeps that step explicit and auditable by adding both an ``addressee`` and an
``addressee_method`` to each turn.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional


BROADCAST_RE = re.compile(
    r"\b(everyone|everybody|all of you|you all|y'all|you guys|guys|team|folks)\b",
    flags=re.IGNORECASE,
)

SECOND_PERSON_RE = re.compile(
    r"\b(you|your|yours|you're|youre|you'll|youll|you've|youve)\b",
    flags=re.IGNORECASE,
)


def infer_addressees(
    turns: Sequence[Mapping],
    players: Optional[Iterable[str]] = None,
    player_aliases: Optional[Mapping[str, Iterable[str]]] = None,
    overwrite: bool = False,
    use_sequential: bool = True,
) -> list[dict]:
    """Return turns with ``addressee`` and ``addressee_method`` fields.

    Inference order:
      1. Preserve human-coded ``receiver_raw`` / ``addressee`` values.
      2. Use visual/facial addressee evidence when present.
      3. Detect broadcast utterances such as "everyone" or "you guys".
      4. Detect direct name/player mentions using provided aliases.
      5. Resolve second-person speech ("you", "your") to nearby context.
      6. Optionally fall back to the next/previous distinct speaker.

    ``addressee_method`` makes weak heuristics visible in downstream review.
    """
    player_list = _player_list(turns, players)
    alias_patterns = _alias_patterns(player_list, player_aliases or {})

    inferred: list[dict] = []
    for idx, turn in enumerate(turns):
        out = dict(turn)
        speaker = str(out.get("speaker", "")).strip()
        utterance = str(out.get("utterance", "") or "")

        coded = str(out.get("receiver_raw", "") or out.get("addressee", "") or "").strip()
        if coded and not overwrite:
            out["addressee"] = normalize_addressee(coded, speaker, player_list)
            out["addressee_method"] = "coded"
            inferred.append(out)
            continue

        visual = str(out.get("visual_addressee", "") or "").strip()
        if visual and not overwrite:
            out["addressee"] = normalize_addressee(visual, speaker, player_list)
            out["addressee_method"] = str(out.get("visual_method", "") or "visual_face_gaze")
            inferred.append(out)
            continue

        addressee = ""
        method = "unknown"

        if BROADCAST_RE.search(utterance) and _other_players(speaker, player_list):
            addressee = "All"
            method = "broadcast_keyword"
        else:
            mentioned = _mentioned_players(utterance, speaker, player_list, alias_patterns)
            if mentioned:
                addressee = _join_players(mentioned)
                method = "name_mention"
            elif SECOND_PERSON_RE.search(utterance):
                contextual = _nearby_speaker(turns, idx, speaker, prefer_previous=True)
                if contextual:
                    addressee = contextual
                    method = "pronoun_context"
            if not addressee and use_sequential:
                contextual = _nearby_speaker(turns, idx, speaker, prefer_previous=False)
                if contextual:
                    addressee = contextual
                    method = "sequential_context"

        out["addressee"] = addressee
        out["addressee_method"] = method
        inferred.append(out)

    return inferred


def normalize_addressee(raw: str, speaker: str, players: Sequence[str]) -> str:
    """Normalize a coded receiver cell to the CSV representation."""
    raw = str(raw or "").strip()
    if not raw:
        return ""

    parsed = _parse_receivers(raw, set(players))
    if "ALL" in parsed:
        return "All"
    if parsed:
        ordered = [player for player in players if player in parsed and player != speaker]
        return _join_players(ordered)
    return raw


def _parse_receivers(raw: str, players: set[str]) -> list[str]:
    low = raw.strip().lower()
    if low in {"self", "none", "other", "unknown", "room", "the room"}:
        return []
    if low == "all":
        return ["ALL"]

    raw = re.sub(r"\?$", "", raw).strip()
    parts = re.split(r"\s+and\s+|[/,;]", raw, flags=re.IGNORECASE)
    normalized_players = {player.upper(): player for player in players}
    receivers = []
    for part in parts:
        key = part.strip().upper()
        if key in normalized_players:
            receivers.append(normalized_players[key])
    return receivers


def _player_list(turns: Sequence[Mapping], players: Optional[Iterable[str]]) -> list[str]:
    if players:
        return [str(player).strip() for player in players if str(player).strip()]

    seen: list[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker", "")).strip()
        if speaker and speaker not in seen:
            seen.append(speaker)
    return sorted(seen)


def _alias_patterns(
    players: Sequence[str],
    player_aliases: Mapping[str, Iterable[str]],
) -> dict[str, list[re.Pattern]]:
    patterns: dict[str, list[re.Pattern]] = {}
    for player in players:
        aliases = list(player_aliases.get(player, []))
        if len(player) == 1:
            aliases.extend([f"player {player}", f"participant {player}", f"person {player}"])
        else:
            aliases.append(player)

        compiled = []
        for alias in aliases:
            alias = str(alias).strip()
            if not alias:
                continue
            compiled.append(
                re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE)
            )
        patterns[player] = compiled
    return patterns


def _mentioned_players(
    utterance: str,
    speaker: str,
    players: Sequence[str],
    alias_patterns: Mapping[str, Sequence[re.Pattern]],
) -> list[str]:
    mentioned = []
    for player in players:
        if player == speaker:
            continue
        if any(pattern.search(utterance) for pattern in alias_patterns.get(player, [])):
            mentioned.append(player)
    return mentioned


def _nearby_speaker(
    turns: Sequence[Mapping],
    idx: int,
    speaker: str,
    prefer_previous: bool,
) -> str:
    directions = ((-1, 1) if prefer_previous else (1, -1))
    for direction in directions:
        cursor = idx + direction
        while 0 <= cursor < len(turns):
            candidate = str(turns[cursor].get("speaker", "")).strip()
            if candidate and candidate != speaker:
                return candidate
            cursor += direction
    return ""


def _other_players(speaker: str, players: Sequence[str]) -> list[str]:
    return [player for player in players if player != speaker]


def _join_players(players: Iterable[str]) -> str:
    return ";".join(str(player) for player in players if str(player))
