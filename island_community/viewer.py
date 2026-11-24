"""Serve recorded simulation state without executing policies or advancing the world."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .config import load_config


class ReplayIndex:
    """Index complete frame offsets so seeking loads one world state rather than the entire run."""

    maximum_frame_bytes = 16_000_000

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = RLock()
        self._identity: tuple[int, int] | None = None
        self._offset = 0
        self._modified = 0
        self._positions: dict[int, tuple[int, int]] = {}
        self._metrics: dict[int, dict[str, Any]] = {}

    def _refresh(self) -> None:
        """Scan newly appended complete lines and discard stale offsets after replacement."""
        source = self.directory / "frames.jsonl"
        if not source.exists():
            self._positions.clear()
            self._metrics.clear()
            self._identity, self._offset = None, 0
            return
        with source.open("rb") as stream:
            information = os.fstat(stream.fileno())
            identity = (information.st_dev, information.st_ino)
            rewritten = information.st_size <= self._offset and information.st_mtime_ns != self._modified
            if identity != self._identity or information.st_size < self._offset or rewritten:
                self._positions.clear()
                self._metrics.clear()
                self._offset = 0
            self._identity, self._modified = identity, information.st_mtime_ns
            stream.seek(self._offset)
            while True:
                start = stream.tell()
                line = stream.readline(self.maximum_frame_bytes + 1)
                if len(line) > self.maximum_frame_bytes:
                    raise ValueError("Replay frame exceeds the viewer's size limit")
                if not line or not line.endswith(b"\n"):
                    break
                frame = json.loads(line)
                assert isinstance(frame, dict)
                turn = frame["turn"]
                if type(turn) is not int or turn < 0 or (self._positions and turn < next(reversed(self._positions))):
                    raise ValueError("Replay turns must be nonnegative and ordered")
                self._positions[turn] = (start, len(line))
                self._metrics[turn] = {"turn": turn, **frame.get("metrics", {})}
                self._offset = stream.tell()

    def frame(self, turn: int | None = None) -> dict[str, Any]:
        """Read an exact historical frame or fall back to a legacy recording's latest snapshot."""
        with self._lock:
            self._refresh()
            if self._positions:
                selected_turn = next(reversed(self._positions)) if turn is None else turn
                if selected_turn not in self._positions:
                    raise KeyError("Turn is not recorded")
                offset, length = self._positions[selected_turn]
                with (self.directory / "frames.jsonl").open("rb") as stream:
                    stream.seek(offset)
                    frame = json.loads(stream.read(length))
            else:
                source = self.directory / "snapshot.json"
                if not source.exists():
                    raise KeyError("No recorded snapshot yet")
                if source.stat().st_size > self.maximum_frame_bytes:
                    raise ValueError("Snapshot exceeds the viewer's size limit")
                frame = json.loads(source.read_bytes())
            assert isinstance(frame, dict)
            if turn is not None and frame["turn"] != turn:
                raise KeyError("Turn is not recorded")
            return frame

    def timeline(self) -> dict[str, Any]:
        """Describe available turns while remaining compatible with old latest-only recordings."""
        with self._lock:
            self._refresh()
            turns = list(self._positions)
            has_history = bool(turns)
            if not turns:
                try:
                    turns = [self.frame()["turn"]]
                except KeyError:
                    pass
            manifest_path = self.directory / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes()) if manifest_path.exists() else {}
            assert isinstance(manifest, dict)
            return {"turns": turns, "has_history": has_history, "status": manifest.get("status", "unknown")}

    def metrics(self, maximum_rows: int = 2000) -> list[dict[str, Any]]:
        """Return chart samples spanning the whole recording without retaining full frames."""
        with self._lock:
            self._refresh()
            rows = list(self._metrics.values())
            if not rows:
                return read_metrics(self.directory / "metrics.jsonl", maximum_rows)
            if len(rows) <= maximum_rows:
                return rows
            return [rows[index * (len(rows) - 1) // (maximum_rows - 1)] for index in range(maximum_rows)]


def read_metrics(path: Path, maximum_rows: int = 2000) -> list[dict[str, Any]]:
    """Read a bounded tail of complete metric records while a run is appending data."""
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        offset = max(0, stream.tell() - 2_000_000)
        stream.seek(offset)
        if offset:
            stream.readline()
        content = stream.read()
    complete_lines = content.split(b"\n")[:-1]
    return [json.loads(line) for line in complete_lines[-maximum_rows:] if line.strip()]


def create_server(directory: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create a read-only viewer that seeks recorded frames without executing the world."""
    run_directory = directory.resolve()
    page = Path(__file__).with_name("viewer.html").read_bytes()
    replay = ReplayIndex(run_directory)

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            address = urlsplit(self.path)
            path = address.path
            try:
                if path == "/":
                    body, content_type = page, "text/html; charset=utf-8"
                elif path == "/api/snapshot":
                    parameters = parse_qs(address.query, keep_blank_values=True)
                    requested = parameters.get("turn")
                    if requested is not None and (len(requested) != 1 or not requested[0].isascii() or not requested[0].isdigit() or len(requested[0]) > 12):
                        self.respond(400, b'{"error":"Turn must be a nonnegative integer"}', "application/json")
                        return
                    snapshot = replay.frame(int(requested[0]) if requested is not None else None)
                    body, content_type = json.dumps(snapshot, allow_nan=False).encode(), "application/json"
                elif path == "/api/timeline":
                    body, content_type = json.dumps(replay.timeline()).encode(), "application/json"
                elif path == "/api/decision":
                    parameters = parse_qs(address.query, keep_blank_values=True)
                    if any(len(parameters.get(key, [])) != 1 or not parameters[key][0].isascii()
                           or not parameters[key][0].isdigit() or len(parameters[key][0]) > 12 for key in ("turn", "island")):
                        self.respond(400, b'{"error":"Turn and island must be nonnegative integers"}', "application/json")
                        return
                    turn, island_id = int(parameters["turn"][0]), int(parameters["island"][0])
                    scores_path = run_directory / "decisions" / f"{turn}.json"
                    if not scores_path.exists():
                        self.respond(404, b'{"error":"Policy scores were not recorded for this turn"}', "application/json")
                        return
                    if scores_path.stat().st_size > 64_000_000:
                        raise ValueError("Policy score record exceeds the viewer's size limit")
                    scores = json.loads(scores_path.read_bytes())
                    assert isinstance(scores, dict)
                    body = json.dumps({"turn": turn, "island_id": island_id, "stages": scores["islands"][str(island_id)]}, allow_nan=False).encode()
                    content_type = "application/json"
                elif path == "/api/metrics":
                    body = json.dumps(replay.metrics(), allow_nan=False).encode()
                    content_type = "application/json"
                elif path == "/api/rules":
                    configuration_path = run_directory / "config.json"
                    recorded = json.loads(configuration_path.read_text()) if configuration_path.exists() else None
                    body = json.dumps({"recorded": recorded, "defaults": load_config()}, allow_nan=False).encode()
                    content_type = "application/json"
                else:
                    self.respond(404, b'{"error":"Unknown endpoint"}', "application/json")
                    return
                self.respond(200, body, content_type)
            except KeyError:
                self.respond(404, b'{"error":"Requested turn is not recorded"}', "application/json")
            except (OSError, ValueError, TypeError, AssertionError):
                self.respond(503, b'{"error":"Recorded data is temporarily unavailable"}', "application/json")

        def respond(self, status: int, body: bytes, content_type: str) -> None:
            """Write a complete response with explicit cache and content-type behavior."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *arguments: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), ViewerHandler)


def serve(directory: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Keep the local inspector available until the user stops its server."""
    server = create_server(directory, host, port)
    print(f"Island Community viewer: http://{host}:{server.server_port}", flush=True)
    print(f"Reading recorded state from {directory.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
