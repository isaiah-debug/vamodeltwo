"""Visual addressee evidence from participant faces.

This module is intentionally separate from transcription. Audio tracks identify
the speaker; video frames help estimate who that speaker is addressing.

The heavy path uses OpenCV + InsightFace only when called. The pure scoring
functions are testable without model downloads.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def apply_visual_addressees(
    turns: Sequence[Mapping],
    observations_by_turn: Mapping[int, Sequence[Mapping]],
    yaw_threshold: float = 12.0,
    min_votes: int = 1,
) -> list[dict]:
    """Attach visual addressee fields using precomputed face observations.

    ``observations_by_turn`` is keyed by turn number and contains detections
    with at least ``player``, ``bbox``, and optionally ``yaw`` and
    ``confidence``.
    """
    enriched = []
    for turn in turns:
        out = dict(turn)
        observations = observations_by_turn.get(int(out.get("turn", 0)), [])
        selection = select_visual_addressee(
            speaker=str(out.get("speaker", "")),
            observations=observations,
            yaw_threshold=yaw_threshold,
            min_votes=min_votes,
        )
        out.update(selection)
        enriched.append(out)
    return enriched


def select_visual_addressee(
    speaker: str,
    observations: Sequence[Mapping],
    yaw_threshold: float = 12.0,
    min_votes: int = 1,
) -> dict:
    """Select a likely addressee from face detections in one utterance window."""
    frame_groups = _group_observations_by_frame(observations)
    if not frame_groups:
        return _empty_visual("no_visual_match")

    votes: Counter[str] = Counter()
    confidence_totals: defaultdict[str, float] = defaultdict(float)
    evidence = []
    for frame_observations in frame_groups:
        speaker_faces = [
            obs for obs in frame_observations
            if str(obs.get("player", "")) == speaker and _bbox(obs)
        ]
        other_faces = [
            obs for obs in frame_observations
            if str(obs.get("player", "")) and str(obs.get("player", "")) != speaker and _bbox(obs)
        ]
        for speaker_face in speaker_faces:
            target = _target_for_speaker_face(
                speaker_face=speaker_face,
                candidates=other_faces,
                yaw_threshold=yaw_threshold,
            )
            if not target:
                continue
            player = str(target.get("player"))
            confidence = float(target.get("confidence", 1.0) or 1.0)
            votes[player] += 1
            confidence_totals[player] += confidence
            evidence.append(
                {
                    "timestamp_s": speaker_face.get("timestamp_s", ""),
                    "speaker_yaw": _yaw(speaker_face),
                    "target": player,
                    "target_confidence": round(confidence, 3),
                }
            )

    if not votes:
        return _empty_visual("no_gaze_target")

    winner, vote_count = votes.most_common(1)[0]
    if vote_count < min_votes:
        return _empty_visual("insufficient_visual_votes")

    mean_confidence = confidence_totals[winner] / vote_count
    return {
        "visual_addressee": winner,
        "visual_confidence": round(mean_confidence, 3),
        "visual_method": "face_gaze",
        "visual_votes": vote_count,
        "visual_evidence": json.dumps(evidence, ensure_ascii=False),
    }


def infer_visual_addressees(
    turns: Sequence[Mapping],
    media_paths: Sequence[str | Path],
    face_references: Mapping[str, str | Path],
    out_dir: str | Path,
    sample_fps: float = 1.0,
    identity_threshold: float = 0.35,
    yaw_threshold: float = 12.0,
) -> list[dict]:
    """Run optional InsightFace analysis and attach visual addressee evidence."""
    if not face_references:
        raise ValueError(
            "visual addressee inference requires face reference images, e.g. "
            "A=data/faces/A.jpg B=data/faces/B.jpg"
        )

    detector = InsightFaceDetector(face_references, identity_threshold=identity_threshold)
    media_lookup = {Path(path).name: Path(path) for path in media_paths}
    observations_by_turn: dict[int, list[dict]] = {}

    for turn in turns:
        source_file = str(turn.get("source_file", ""))
        media_path = media_lookup.get(source_file)
        if not media_path:
            continue
        observations_by_turn[int(turn["turn"])] = detector.observations_for_turn(
            media_path=media_path,
            start_s=float(turn.get("local_start_s", turn.get("start_s", 0.0))),
            end_s=float(turn.get("local_end_s", turn.get("end_s", 0.0))),
            sample_fps=sample_fps,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "visual_observations.json").open("w", encoding="utf-8") as handle:
        json.dump(observations_by_turn, handle, indent=2, ensure_ascii=False)

    return apply_visual_addressees(
        turns,
        observations_by_turn,
        yaw_threshold=yaw_threshold,
    )


class InsightFaceDetector:
    """Small wrapper around InsightFace for face ID and pose observations."""

    def __init__(
        self,
        face_references: Mapping[str, str | Path],
        identity_threshold: float = 0.35,
    ) -> None:
        self.identity_threshold = identity_threshold
        self.app = _load_face_app()
        self.references = self._load_references(face_references)

    def observations_for_turn(
        self,
        media_path: str | Path,
        start_s: float,
        end_s: float,
        sample_fps: float,
    ) -> list[dict]:
        import cv2

        media_path = Path(media_path)
        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for visual analysis: {media_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step_s = 1.0 / sample_fps if sample_fps > 0 else 1.0
        observations = []
        t = max(0.0, start_s)
        while t <= end_s:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            observations.extend(self._observations_for_frame(frame, t))
            t += step_s

        cap.release()
        return observations

    def _load_references(self, face_references: Mapping[str, str | Path]) -> dict[str, list[float]]:
        import cv2

        references = {}
        for player, image_path in face_references.items():
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Face reference not found or unreadable: {image_path}")
            faces = self.app.get(image)
            if not faces:
                raise ValueError(f"No face detected in reference image for {player}: {image_path}")
            face = max(faces, key=lambda item: _area(item.bbox))
            references[str(player)] = _normalize(face.embedding)
        return references

    def _observations_for_frame(self, frame, timestamp_s: float) -> list[dict]:
        observations = []
        for face in self.app.get(frame):
            player, confidence = self._match_player(face.embedding)
            if not player:
                continue
            bbox = [float(value) for value in face.bbox.tolist()]
            observations.append(
                {
                    "timestamp_s": round(timestamp_s, 3),
                    "player": player,
                    "confidence": round(confidence, 3),
                    "bbox": bbox,
                    "yaw": _face_yaw(face),
                }
            )
        return observations

    def _match_player(self, embedding) -> tuple[str, float]:
        embedding = _normalize(embedding)
        best_player = ""
        best_score = -1.0
        for player, reference in self.references.items():
            score = _cosine(embedding, reference)
            if score > best_score:
                best_player = player
                best_score = score
        if best_score < self.identity_threshold:
            return "", best_score
        return best_player, best_score


def _target_for_speaker_face(
    speaker_face: Mapping,
    candidates: Sequence[Mapping],
    yaw_threshold: float,
) -> Mapping | None:
    speaker_center = _center(_bbox(speaker_face))
    yaw = _yaw(speaker_face)
    direction = 0
    if yaw <= -yaw_threshold:
        direction = -1
    elif yaw >= yaw_threshold:
        direction = 1

    directional_candidates = []
    for candidate in candidates:
        candidate_center = _center(_bbox(candidate))
        dx = candidate_center[0] - speaker_center[0]
        if direction == 0 or (direction < 0 and dx < 0) or (direction > 0 and dx > 0):
            directional_candidates.append(candidate)

    if not directional_candidates:
        return None

    return min(
        directional_candidates,
        key=lambda candidate: _distance(speaker_center, _center(_bbox(candidate))),
    )


def _group_observations_by_frame(observations: Sequence[Mapping]) -> list[list[Mapping]]:
    if not observations:
        return []
    if not any("timestamp_s" in observation for observation in observations):
        return [list(observations)]

    grouped: defaultdict[float, list[Mapping]] = defaultdict(list)
    for observation in observations:
        try:
            timestamp = round(float(observation.get("timestamp_s", 0.0)), 3)
        except (TypeError, ValueError):
            timestamp = 0.0
        grouped[timestamp].append(observation)
    return [grouped[key] for key in sorted(grouped)]


def _load_face_app():
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "visual analysis requires insightface. Install with: "
            "python3 -m pip install -r requirements-video.txt"
        ) from exc

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _empty_visual(method: str) -> dict:
    return {
        "visual_addressee": "",
        "visual_confidence": "",
        "visual_method": method,
        "visual_votes": 0,
        "visual_evidence": "[]",
    }


def _bbox(observation: Mapping) -> list[float]:
    bbox = observation.get("bbox") or []
    if len(bbox) != 4:
        return []
    return [float(value) for value in bbox]


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _yaw(observation: Mapping) -> float:
    try:
        return float(observation.get("yaw", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _face_yaw(face) -> float:
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) < 2:
        return 0.0
    return float(pose[1])


def _area(bbox) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def _normalize(vector) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
