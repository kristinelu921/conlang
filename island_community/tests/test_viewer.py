"""Verify bounded artifact reading and read-only viewer endpoints."""

import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from island_community.viewer import ReplayIndex, create_server, read_metrics


def test_metric_reader_ignores_incomplete_last_record_and_bounds_rows(tmp_path: Path) -> None:
    source = tmp_path / "metrics.jsonl"
    source.write_text('{"turn":1}\n{"turn":2}\n{"turn":')
    assert read_metrics(source, maximum_rows=1) == [{"turn": 2}]
    assert read_metrics(tmp_path / "absent") == []


def test_viewer_exposes_only_recorded_artifacts(tmp_path: Path) -> None:
    (tmp_path / "snapshot.json").write_text('{"turn":7,"islands":[]}')
    (tmp_path / "secret.txt").write_text("not served")
    server = create_server(tmp_path, port=0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(address + "/api/snapshot", timeout=3) as response:
            assert json.load(response)["turn"] == 7
        with urlopen(address + "/", timeout=3) as response:
            assert b"The observatory" in response.read()
        with urlopen(address + "/api/timeline", timeout=3) as response:
            assert json.load(response) == {"turns": [7], "has_history": False, "status": "unknown"}
        with urlopen(address + "/api/rules", timeout=3) as response:
            rules = json.load(response)
            assert rules["recorded"] is None
            assert rules["defaults"]["item_utility"]["uses"]["tool"]["bonuses"]["productivity"] == 0.5
        original_rules = {"schema_version": 1, "items": {"grain": {"survival": {"food": 1}}}}
        (tmp_path / "config.json").write_text(json.dumps(original_rules))
        with urlopen(address + "/api/rules", timeout=3) as response:
            rules = json.load(response)
            assert rules["recorded"] == original_rules
            assert rules["defaults"]["schema_version"] == 6
        with pytest.raises(HTTPError) as failure:
            urlopen(address + "/api/snapshot?turn=6", timeout=3)
        assert failure.value.code == 404
        for query in ("-1", "bad", "", "1&turn=2"):
            with pytest.raises(HTTPError) as failure:
                urlopen(address + "/api/snapshot?turn=" + query, timeout=3)
            assert failure.value.code == 400
        for path in ("/secret.txt", "/../secret.txt", "/api/../secret.txt"):
            with pytest.raises(HTTPError) as failure:
                urlopen(address + path, timeout=3)
            assert failure.value.code == 404
        (tmp_path / "snapshot.json").unlink()
        with pytest.raises(HTTPError) as failure:
            urlopen(address + "/api/snapshot", timeout=3)
        assert failure.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_frame_index_seeks_history_and_waits_for_complete_appends(tmp_path: Path) -> None:
    frames = tmp_path / "frames.jsonl"
    frames.write_text('{"turn":2,"metrics":{"population":5}}\n{"turn":3')
    replay = ReplayIndex(tmp_path)
    assert replay.timeline()["turns"] == [2]
    assert replay.frame(2)["metrics"]["population"] == 5
    with pytest.raises(KeyError):
        replay.frame(3)
    with frames.open("a") as stream:
        stream.write(',"metrics":{"population":6}}\n')
    assert replay.timeline()["turns"] == [2, 3]
    assert replay.frame()["turn"] == 3
    assert replay.frame(2)["metrics"]["population"] == 5
    assert replay.metrics() == [{"turn": 2, "population": 5}, {"turn": 3, "population": 6}]


def test_decision_endpoint_selects_one_island_and_rejects_unsafe_or_missing_queries(tmp_path: Path) -> None:
    """Expose saved decision scores with bounded numeric lookups and clear legacy behavior."""
    (tmp_path / "decisions").mkdir()
    stages = {"action": {"temperature": 2, "selected_index": 0, "options": [{"raw_logit": 3, "probability": 1}]}}
    (tmp_path / "decisions" / "3.json").write_text(json.dumps({"turn": 3, "islands": {"7": stages, "8": {"private": "not requested"}}}))
    server = create_server(tmp_path, port=0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}/api/decision"
    try:
        with urlopen(address + "?turn=3&island=7", timeout=3) as response:
            assert json.load(response) == {"turn": 3, "island_id": 7, "stages": stages}
        for query in ("?turn=2&island=7", "?turn=3&island=9"):
            with pytest.raises(HTTPError) as failure:
                urlopen(address + query, timeout=3)
            assert failure.value.code == 404
        for query in ("", "?turn=3", "?turn=3&island=-1", "?turn=../3&island=7", "?turn=3&island=7&island=8"):
            with pytest.raises(HTTPError) as failure:
                urlopen(address + query, timeout=3)
            assert failure.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_frame_index_handles_replacement_duplicate_turns_and_chart_sampling(tmp_path: Path) -> None:
    frames = tmp_path / "frames.jsonl"
    frames.write_text(''.join(json.dumps({"turn": turn, "metrics": {"population": turn}}) + '\n' for turn in range(7)))
    replay = ReplayIndex(tmp_path)
    assert [row["turn"] for row in replay.metrics(maximum_rows=3)] == [0, 3, 6]
    with frames.open("a") as stream:
        stream.write('{"turn":6,"metrics":{"population":99}}\n')
    assert replay.frame(6)["metrics"]["population"] == 99
    assert len(replay.timeline()["turns"]) == 7
    replacement = tmp_path / "replacement"
    replacement.write_text('{"turn":10,"metrics":{"population":1}}\n')
    replacement.replace(frames)
    assert replay.timeline()["turns"] == [10]
    assert replay.frame()["turn"] == 10
    with pytest.raises(KeyError):
        replay.frame(6)


def test_viewer_serves_selected_frames_instead_of_only_final_snapshot(tmp_path: Path) -> None:
    (tmp_path / "frames.jsonl").write_text('{"turn":0,"islands":[{"id":0}]}\n{"turn":1,"islands":[]}\n')
    server = create_server(tmp_path, port=0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(address + "/api/snapshot?turn=0", timeout=3) as response:
            assert json.load(response)["islands"] == [{"id": 0}]
        with urlopen(address + "/api/snapshot", timeout=3) as response:
            assert json.load(response) == {"turn": 1, "islands": []}
        with urlopen(address + "/api/timeline", timeout=3) as response:
            assert json.load(response)["turns"] == [0, 1]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
