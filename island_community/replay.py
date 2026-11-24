"""Reconstruct legacy recordings from recorded choices without executing neural policies."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

from island_community.config import load_config
from island_community.storage import playback_frame, write_json_atomic
from island_community.world import Action, World


def read_experiences(path: Path) -> Iterator[dict[str, Any]]:
    """Stream complete transition records and reject truncated input."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("The experience journal ends with an incomplete record")
            record = json.loads(line)
            assert isinstance(record, dict)
            yield record


def rebuild_replay(recording: Path, output: Path) -> Path:
    """Verify and reconstruct a completed founding run on authorized compute into a new directory."""
    recording, output = recording.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Replay output already exists: {output}")
    manifest = json.loads((recording / "manifest.json").read_text())
    assert isinstance(manifest, dict)
    if manifest["status"] not in {"completed", "extinct"}:
        raise ValueError("Finish the recorded run before rebuilding its replay")
    if manifest.get("resumed_from") is not None:
        raise ValueError("Legacy continuation segments need their starting state; new runs record replay frames automatically")
    source_names = ["world.py", "config.py"]
    if manifest["config"]["schema_version"] >= 8:
        source_names.append("network.py")
    for name in source_names:
        expected = manifest.get("source_sha256", {}).get(name)
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Replay needs the original {name}; use the recording's matching source version")
    configuration = load_config(recording / "config.json")
    world = World(configuration)
    final_snapshot = json.loads((recording / "snapshot.json").read_text())
    assert isinstance(final_snapshot, dict)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as temporary_name:
        staging = Path(temporary_name)
        with (staging / "frames.jsonl").open("w", encoding="utf-8") as frames:
            snapshot = world.snapshot()
            snapshot["maximum_health"] = configuration["survival"]["maximum_health"]
            frame = playback_frame(snapshot, [])
            frames.write(json.dumps(frame, allow_nan=False) + "\n")
            for turn, transitions in groupby(read_experiences(recording / "experiences.jsonl"), key=lambda record: record["turn"]):
                records = list(transitions)
                if turn != world.turn + 1 or not world.islands:
                    raise ValueError(f"Experience history is not contiguous at turn {turn}")
                decisions = world.start_turn()
                by_island = {record["island_id"]: record for record in records}
                if len(by_island) != len(records) or by_island.keys() != decisions.keys():
                    raise ValueError(f"Missing or duplicate island decisions at turn {turn}")
                for island_id, decision in decisions.items():
                    if decision.observation != by_island[island_id]["observation"]:
                        raise ValueError(f"Observation mismatch at turn {turn}, island {island_id}")
                actions = {island_id: Action(**record["action"]) for island_id, record in by_island.items()}
                for island_id, record in by_island.items():
                    if record.get("schema_version") == 2:
                        template = decisions[island_id].candidates[record["candidate_index"]]
                        quantity_candidates = world.quantity_choices(island_id, template)
                        if (len(quantity_candidates) != record["quantity_candidate_count"]
                                or quantity_candidates[record["quantity_index"]] != actions[island_id]
                                or world._action_features(island_id, actions[island_id]) != record["action_features"]):
                            raise ValueError(f"Quantity choice mismatch at turn {turn}, island {island_id}")
                outcome = world.finish_turn(actions)
                next_observations = world.preview_next_observations() if world.islands else {}
                for island_id, record in by_island.items():
                    if record["reward"] != outcome.rewards[island_id] or record["terminal"] != (island_id in outcome.deaths):
                        raise ValueError(f"Outcome mismatch at turn {turn}, island {island_id}")
                    if "reward_components" in record and record["reward_components"] != outcome.reward_components.get(island_id):
                        raise ValueError(f"Reward breakdown mismatch at turn {turn}, island {island_id}")
                    if record["next_observation"] != next_observations.get(island_id):
                        raise ValueError(f"Next observation mismatch at turn {turn}, island {island_id}")
                snapshot = world.snapshot()
                snapshot["maximum_health"] = configuration["survival"]["maximum_health"]
                frame = playback_frame(snapshot, records)
                frames.write(json.dumps(frame, allow_nan=False) + "\n")
        normalized = json.loads(json.dumps(snapshot))
        if world.turn != manifest["final_turn"] or any(final_snapshot.get(key) != value for key, value in normalized.items()):
            raise ValueError("Reconstruction does not match the original final world snapshot")
        write_json_atomic(staging / "snapshot.json", frame)
        shutil.copyfile(recording / "metrics.jsonl", staging / "metrics.jsonl")
        shutil.copyfile(recording / "config.json", staging / "config.json")
        write_json_atomic(staging / "manifest.json", {
            "schema_version": 1,
            "recording_format": "per_turn_frames_v1",
            "kind": "verified_reconstruction",
            "source_recording": str(recording),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": manifest["status"],
            "final_turn": world.turn,
            "frame_count": world.turn + 1,
            "source_sha256": manifest["source_sha256"],
            "recording_sha256": {
                name: hashlib.sha256((recording / name).read_bytes()).hexdigest()
                for name in ("config.json", "experiences.jsonl", "snapshot.json", "manifest.json")
            },
        })
        if output.exists():
            raise FileExistsError(f"Replay output already exists: {output}")
        staging.rename(output)
    return output
