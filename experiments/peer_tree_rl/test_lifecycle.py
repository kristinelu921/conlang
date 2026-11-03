"""Remote tests for durable evidence, resume integrity, and allocation budget boundaries."""

import unittest
from pathlib import Path

if not Path(__file__).resolve().is_relative_to("/home/kristinelu/peer-tree"):
    raise unittest.SkipTest("Pilot tests execute only on authorized remote compute.")

import json
import shutil
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

from run import Study, verify_checkpoint, write_json, require_remote
from protocol import sample_seed


class FixtureBackend:
    def checkpoint(self, directory: Path) -> None:
        """Write small stand-ins for the real separately tested optimizer serialization."""
        directory.mkdir(parents=True)
        for agent in range(2):
            (directory / f"agent-{agent}.msgpack").write_bytes(bytes([agent, 17, 29]))
        write_json(directory / "configuration.json", {"model_revision": "fixture"})


class LifecycleTests(unittest.TestCase):
    def test_failure_archive_contains_evidence_without_an_optimizer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "attempt"
            output.mkdir()
            study = Study({}, {"source_commit": "source", "gcs_prefix": "gs://fixture/run"}, output)
            study.wandb_run = SimpleNamespace(summary={})
            write_json(output / "terminal.json", {"status": "failed"})
            write_json(output / "tree-0000.json", {"tokens": [17], "reward": 0})
            destination = root / "cloud"
            destination.mkdir()

            def copy_command(command: list[str], **kwargs: object) -> None:
                for name in command[3:-1]:
                    shutil.copyfile(name, destination / Path(name).name)

            with patch("run.subprocess.run", side_effect=copy_command), patch("run.subprocess.check_output",
                    side_effect=lambda command: (destination / "failure-manifest.json").read_bytes()):
                study.archive_failure()
            self.assertEqual(json.loads((destination / "terminal.json").read_text())["status"], "failed")
            self.assertTrue((destination / "tree-0000.json").exists())
            self.assertFalse((destination / "agent-0.msgpack").exists())

    def test_durable_copy_restores_complete_evidence_and_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "attempt"
            output.mkdir()
            settings = {"model_revision": "fixture"}
            receipt = {"deadline_unix_seconds": 1234, "source_commit": "source", "gcs_prefix": "gs://fixture/run"}
            study = Study(settings, receipt, output)
            study.backend = FixtureBackend()
            study.completed_trees = 7
            write_json(output / "train.json", [{"problem": "original", "answer": 42}])
            write_json(output / "tree-0006.json", {"tokens": [17, 29], "reward": 1})
            (output / "events.jsonl").write_text('{"event": "updated"}\n')
            destination = root / "cloud" / "tree-0006"

            def copy_command(command: list[str], **kwargs: object) -> None:
                shutil.copytree(Path(command[-2]), destination)

            def read_command(command: list[str]) -> bytes:
                if command[2] == "cat":
                    return (destination / "complete.json").read_bytes()
                return str((destination / command[-2].rsplit("/", 1)[-1]).stat().st_size).encode()

            with patch("run.subprocess.run", side_effect=copy_command), patch("run.subprocess.check_output", side_effect=read_command):
                study.save_checkpoint("tree-0006")
            shutil.rmtree(output)
            manifest = verify_checkpoint(destination, settings, receipt)
            self.assertEqual(manifest["completed_trees"], 7)
            self.assertFalse(manifest["training_finished"])
            self.assertEqual(json.loads((destination / "evidence/tree-0006.json").read_text())["tokens"], [17, 29])
            self.assertTrue((destination / "evidence/events.jsonl").exists())
            self.assertEqual(sample_seed(17, "tree", manifest["completed_trees"]), sample_seed(17, "tree", 7))
            with self.assertRaises(ValueError):
                verify_checkpoint(destination, settings, {**receipt, "deadline_unix_seconds": 9999})
            (destination / "evidence/train.json").write_text("corrupted")
            with self.assertRaises(ValueError):
                verify_checkpoint(destination, settings, receipt)

    def test_remote_guard_rejects_development_host_and_budget_extension(self) -> None:
        receipt = {"remote_hostname": "remote", "control_hostname": "control", "tpu_name": "kristine-fixture",
                   "tpu_type": "v5litepod-8", "spot": True, "zone": "us-west4-a", "region": "us-west4",
                   "allocation_created_unix_seconds": 1000, "deadline_unix_seconds": 87400, "source_sha256": {}}
        with patch("run.socket.gethostname", return_value="control"), self.assertRaises(PermissionError):
            require_remote(receipt, Path("/home/kristinelu/peer-tree/source"))
        with patch("run.socket.gethostname", return_value="remote"), patch("run.time.time", return_value=1001):
            require_remote(receipt, Path("/home/kristinelu/peer-tree/source"))
            with self.assertRaises(ValueError):
                require_remote({**receipt, "deadline_unix_seconds": 87401}, Path("/home/kristinelu/peer-tree/source"))


if __name__ == "__main__":
    unittest.main()
