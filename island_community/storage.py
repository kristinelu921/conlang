"""Atomic run artifacts and complete, versioned simulation checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: Any) -> None:
    """Publish complete JSON files so readers never observe partial snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            # Registry order determines feature layout and seeded sampling order.
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_checkpoint(path: Path, config: dict[str, Any], world_state: dict[str, Any], policy_state: dict[str, Any] | None) -> None:
    """Atomically save world, configuration, and independent learner continuation state."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save({"schema_version": 1, "config": config, "world": world_state, "policies": policy_state}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load tensor and primitive checkpoint data without enabling arbitrary pickle objects."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported island checkpoint schema")
    if not {"config", "world", "policies"} <= state.keys():
        raise ValueError("Checkpoint is missing configuration, world, or policy state")
    return state


class RunRecorder:
    """Record a new immutable run segment without overwriting an earlier experiment."""

    def __init__(self, directory: Path, config: dict[str, Any], command: list[str], resumed_from: Path | None = None) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.event_count = 0
        self.metadata: dict[str, Any] = {
            "schema_version": 1,
            "recording_format": "per_turn_frames_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "config": config,
            "resumed_from": str(resumed_from.resolve()) if resumed_from else None,
            "status": "running",
            "source_sha256": {},
        }
        source_directory = directory / "source"
        source_directory.mkdir()
        for source_path in sorted(Path(__file__).parent.iterdir()):
            if source_path.is_file() and source_path.suffix in {".py", ".json", ".html", ".txt", ".md"}:
                source_bytes = source_path.read_bytes()
                self.metadata["source_sha256"][source_path.name] = hashlib.sha256(source_bytes).hexdigest()
                (source_directory / source_path.name).write_bytes(source_bytes)
        write_json_atomic(directory / "config.json", config)
        write_json_atomic(directory / "manifest.json", self.metadata)

    def record(
        self,
        snapshot: dict[str, Any],
        events: list[dict[str, Any]],
        learning: dict[str, float],
        experiences: list[dict[str, Any]] | None = None,
        policy_scores: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        """Append outcomes, accounting changes, and each acting island's private experience."""
        metrics = {"turn": snapshot["turn"], **snapshot.get("metrics", {}), "learning": learning}
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n")
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            for event in events[self.event_count:]:
                stream.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
        self.event_count = len(events)
        with (self.directory / "experiences.jsonl").open("a", encoding="utf-8") as stream:
            for experience in experiences or []:
                stream.write(json.dumps(experience, allow_nan=False, sort_keys=True) + "\n")
        if policy_scores:
            write_json_atomic(self.directory / "decisions" / f"{snapshot['turn']}.json", {
                "turn": snapshot["turn"], "islands": policy_scores,
            })
        frame = playback_frame(snapshot, experiences or [])
        with (self.directory / "frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(frame, allow_nan=False) + "\n")
        write_json_atomic(self.directory / "snapshot.json", frame)

    def finish(self, status: str, turn: int, error: str | None = None) -> None:
        """Finalize the segment status independently from the world's extinction state."""
        self.metadata.update(status=status, final_turn=turn, finished_at=datetime.now(timezone.utc).isoformat())
        if error is not None:
            self.metadata["error"] = error
        write_json_atomic(self.directory / "manifest.json", self.metadata)


def playback_frame(snapshot: dict[str, Any], experiences: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach inspectable choices without duplicating full neural observations in every frame."""
    return {
        **snapshot,
        "decisions": [
            {**{key: experience[key] for key in ("island_id", "action", "reward", "terminal")},
             **{key: experience[key] for key in ("candidate_counts", "quantity_candidate_count", "action_results") if key in experience},
             **({"reward_components": experience["reward_components"]} if "reward_components" in experience else {})}
            for experience in experiences
        ],
    }
