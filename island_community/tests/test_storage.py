"""Artifact checks that do not execute simulation or neural workloads."""

import json
from pathlib import Path

import pytest

from island_community.storage import RunRecorder, write_json_atomic


def test_atomic_json_retains_previous_snapshot_on_encoding_failure(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot.json"
    write_json_atomic(destination, {"turn": 2})
    with pytest.raises(ValueError):
        write_json_atomic(destination, {"turn": float("nan")})
    assert json.loads(destination.read_text()) == {"turn": 2}
    assert list(tmp_path.iterdir()) == [destination]


def test_saved_configuration_preserves_registry_order(tmp_path: Path) -> None:
    """Saving a resolved configuration must preserve feature and sampling order."""
    from island_community.config import load_config
    from island_community.world import World

    configuration = load_config()
    recorder = RunRecorder(tmp_path / "run", configuration, ["island-community"])
    recorded_configuration = load_config(recorder.directory / "config.json")
    assert list(recorded_configuration["items"]) == list(configuration["items"])
    assert World(recorded_configuration).state_dict() == World(configuration).state_dict()


def test_recorder_appends_only_new_events_and_never_overwrites_run(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    recorder = RunRecorder(directory, {"seed": 11}, ["python", "-m", "island_community"])
    events = [{"kind": "allocation"}]
    recorder.record({"turn": 1, "metrics": {"population": 5}}, events, {})
    events.append({"kind": "birth"})
    recorder.record({"turn": 2, "metrics": {"population": 6}}, events, {"loss": 0.5})
    recorder.finish("completed", 2)
    assert len((directory / "events.jsonl").read_text().splitlines()) == 2
    assert len((directory / "metrics.jsonl").read_text().splitlines()) == 2
    frames = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    assert [frame["turn"] for frame in frames] == [1, 2]
    assert [frame["metrics"]["population"] for frame in frames] == [5, 6]
    assert json.loads((directory / "snapshot.json").read_text()) == frames[-1]
    assert json.loads((directory / "manifest.json").read_text())["status"] == "completed"
    assert (directory / "source" / "storage.py").exists()
    with pytest.raises(FileExistsError):
        RunRecorder(directory, {}, [])


def test_recorder_preserves_individual_wait_and_terminal_experiences(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    recorder = RunRecorder(directory, {"seed": 11}, ["python", "-m", "island_community"])
    first = {
        "island_id": 7, "turn": 1, "observation": [0.5, 1.0],
        "action": {"kind": "wait", "args": {}}, "reward": 2.0,
        "terminal": False, "next_observation": [0.4, 1.0],
    }
    terminal = {
        "island_id": 7, "turn": 2, "observation": [0.4, 1.0],
        "action": {"kind": "wait", "args": {}}, "reward": 0.0,
        "terminal": True, "next_observation": None,
    }
    recorder.record({"turn": 1}, [], {}, [first])
    prefix = (directory / "experiences.jsonl").read_text()
    recorder.record({"turn": 2}, [], {}, [terminal])
    recorder.record({"turn": 2}, [], {})
    history = (directory / "experiences.jsonl").read_text()
    assert history.startswith(prefix)
    assert [json.loads(line) for line in history.splitlines()] == [first, terminal]
    frames = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    assert frames[0]["decisions"] == [{key: first[key] for key in ("island_id", "action", "reward", "terminal")}]
    assert frames[1]["decisions"][0]["terminal"] is True
    assert "observation" not in frames[0]["decisions"][0]


def test_policy_scores_are_saved_separately_with_counts_in_small_replay_frames(tmp_path: Path) -> None:
    """Keep full score tables available without loading all options into every playback frame."""
    recorder = RunRecorder(tmp_path / "run", {"seed": 11}, ["fixture"])
    action = {"kind": "wait", "args": {}}
    scores = {7: {"action": {"temperature": 1.0, "selected_index": 0,
        "options": [{"action": action, "raw_logit": 2.0, "adjusted_logit": 2.0, "probability": 1.0, "group_log_count": 0.0}]}}}
    experience = {"island_id": 7, "turn": 1, "action": action, "reward": 1.0, "terminal": False,
                  "candidate_counts": {"wait": 1}, "quantity_candidate_count": 1}
    recorder.record({"turn": 1}, [], {}, [experience], scores)
    saved = json.loads((recorder.directory / "decisions" / "1.json").read_text())
    assert saved == {"turn": 1, "islands": {"7": scores[7]}}
    frame = json.loads((recorder.directory / "snapshot.json").read_text())
    assert frame["decisions"][0]["candidate_counts"] == {"wait": 1}
    assert "raw_logit" not in json.dumps(frame)
    assert "raw_logit" not in (recorder.directory / "experiences.jsonl").read_text()


@pytest.mark.parametrize("include_water_penalty", [False, True])
def test_replay_frame_retains_negative_reward_breakdown(include_water_penalty: bool) -> None:
    """Keep the actual learning reward and its survival costs visible in replay decisions."""
    from island_community.storage import playback_frame

    experience = {"island_id": 0, "action": {"kind": "wait", "args": {}}, "reward": -6.0, "terminal": True,
                  "reward_components": {"resource_reward": 0.0, "empty_food_penalty": -5.0,
                                        "health_loss_penalty": -1.0, "total": -6.0,
                                        "health_lost": 2.0, "food_available": 0.0}}
    if include_water_penalty:
        experience["reward"] = -11.0
        experience["reward_components"].update(empty_water_penalty=-5.0, water_available=0.0, total=-11.0)
    frame = playback_frame({"turn": 2}, [experience])
    assert frame["decisions"] == [experience]
