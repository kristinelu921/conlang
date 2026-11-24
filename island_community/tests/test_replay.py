"""Deterministic rule fixtures for verified historical reconstruction, without neural execution."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from island_community.config import load_config
from island_community.replay import rebuild_replay
from island_community.storage import RunRecorder
from island_community.world import World


@pytest.fixture(params=[6, 8])
def recording(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Record two deterministic rule-only turns with real observations and outcomes."""
    configuration = load_config(Path(__file__).parents[1] / "scenarios" / "neighbors_50.json") if request.param == 8 else load_config()
    configuration["events"]["probability_per_turn"] = 0
    world = World(configuration)
    recorder = RunRecorder(tmp_path / "original", configuration, ["test-fixture"])
    snapshot = world.snapshot()
    snapshot["maximum_health"] = configuration["survival"]["maximum_health"]
    recorder.record(snapshot, world.events, {})
    for _ in range(2):
        decisions = world.start_turn()
        actions = {identifier: decision.candidates[0] for identifier, decision in decisions.items()}
        quantity_records = {}
        if request.param == 8:
            for identifier, decision in decisions.items():
                candidate_index = next(index for index, action in enumerate(decision.candidates) if action.kind == "propose_trade")
                quantity_candidates = world.quantity_choices(identifier, decision.candidates[candidate_index])
                quantity_index = min(7, len(quantity_candidates) - 1)
                actions[identifier] = quantity_candidates[quantity_index]
                quantity_records[identifier] = {
                    "schema_version": 2, "candidate_index": candidate_index, "quantity_index": quantity_index,
                    "quantity_candidate_count": len(quantity_candidates),
                    "action_features": world._action_features(identifier, actions[identifier]),
                }
        outcome = world.finish_turn(actions)
        following = world.preview_next_observations()
        experiences = [
            {"island_id": identifier, "turn": world.turn, "observation": decision.observation,
             "action": asdict(actions[identifier]), "reward": outcome.rewards[identifier],
             "reward_components": outcome.reward_components[identifier],
             "terminal": identifier in outcome.deaths, "next_observation": following.get(identifier),
             **quantity_records.get(identifier, {})}
            for identifier, decision in decisions.items()
        ]
        snapshot = world.snapshot()
        snapshot["maximum_health"] = configuration["survival"]["maximum_health"]
        recorder.record(snapshot, world.events, {}, experiences)
    recorder.finish("completed", world.turn)
    return recorder.directory


def test_reconstruction_matches_every_recorded_frame_and_preserves_original(recording: Path, tmp_path: Path) -> None:
    before = {path.name: path.read_bytes() for path in recording.iterdir() if path.is_file()}
    output = rebuild_replay(recording, tmp_path / "replay")
    assert [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()] == [
        json.loads(line) for line in (recording / "frames.jsonl").read_text().splitlines()
    ]
    assert json.loads((output / "manifest.json").read_text())["frame_count"] == 3
    assert (output / "config.json").read_bytes() == (recording / "config.json").read_bytes()
    assert before == {path.name: path.read_bytes() for path in recording.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        rebuild_replay(recording, output)


@pytest.mark.parametrize("corruption", ["observation", "reward", "reward_components", "missing_actor", "truncated", "final_state", "engine", "continuation"])
def test_reconstruction_rejects_inconsistent_history_without_publishing_output(recording: Path, tmp_path: Path, corruption: str) -> None:
    journal = recording / "experiences.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    if corruption == "observation":
        records[0]["observation"][0] += 1
    elif corruption == "reward":
        records[0]["reward"] += 1
    elif corruption == "reward_components":
        records[0]["reward_components"]["health_loss_penalty"] -= 1
    elif corruption == "missing_actor":
        records.pop(0)
    elif corruption == "final_state":
        snapshot_path = recording / "snapshot.json"
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["metrics"]["population"] = 999
        snapshot_path.write_text(json.dumps(snapshot))
    elif corruption in {"engine", "continuation"}:
        manifest_path = recording / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if corruption == "engine":
            manifest["source_sha256"]["world.py"] = "wrong"
        else:
            manifest["resumed_from"] = "previous/checkpoint.pt"
        manifest_path.write_text(json.dumps(manifest))
    journal.write_text(''.join(json.dumps(record) + '\n' for record in records))
    if corruption == "truncated":
        journal.write_text(journal.read_text()[:-1])
    output = tmp_path / "replay"
    with pytest.raises(ValueError):
        rebuild_replay(recording, output)
    assert not output.exists()
