"""Remote tests for durable evidence, resume integrity, and allocation budget boundaries."""

import unittest
from pathlib import Path

if not Path(__file__).resolve().is_relative_to("/home/kristinelu/peer-tree"):
    raise unittest.SkipTest("Pilot tests execute only on authorized remote compute.")

import json
import shutil
import tempfile
import time
from unittest.mock import patch
from types import SimpleNamespace

from run import Study, file_record, prepare_data, verify_checkpoint, write_json, require_remote
from protocol import Generation, Problem, State, sample_seed


class FixtureBackend:
    def checkpoint(self, directory: Path) -> None:
        """Write small stand-ins for the real separately tested optimizer serialization."""
        directory.mkdir(parents=True)
        for agent in range(2):
            (directory / f"agent-{agent}.msgpack").write_bytes(bytes([agent, 17, 29]))
        write_json(directory / "configuration.json", {"model_revision": "fixture"})


class LifecycleTests(unittest.TestCase):
    def test_evaluation_searches_only_terminal_output_and_keeps_strict_scores(self) -> None:
        """Ensure answer mentions neither stop dialogue nor contaminate terminal scoring."""
        class EvaluationBackend:
            batch_size = 8

            def generate(self, states: list[State], seeds: list[int], agent: int) -> list[Generation]:
                """Produce a numeric submission and a dialogue with an incidental intermediate answer."""
                responses = []
                for state in states:
                    if state.problem == "submit two":
                        text = '{"action":"submit","answer":2}'
                    elif state.turn == 0:
                        text = '{"action":"message","message":"42 is an intermediate value","context":""}'
                    else:
                        text = "The final answer is 99."
                    responses.append(Generation((11,), (21,), (-0.3,), text, True))
                return responses

        with tempfile.TemporaryDirectory() as temporary:
            for matching, expected in (("strict", 0), ("integer_search", 1)):
                with self.subTest(matching=matching):
                    output = Path(temporary) / matching
                    output.mkdir()
                    study = Study({"seed": 17, "maximum_messages": 8, "answer_matching": matching},
                                  {"deadline_unix_seconds": time.time() + 3600}, output)
                    study.backend = EvaluationBackend()
                    metrics = study.evaluate([Problem("numeric", "submit two", 2, "validation"),
                        Problem("incidental", "original question includes 42", 42, "validation")], "fixture")
                    records = [json.loads(line) for line in (output / "fixture.jsonl").read_text().splitlines()]
                    self.assertEqual(metrics["correct"], expected)
                    self.assertEqual(metrics["strict_correct"], 0)
                    self.assertEqual(records[0]["correct"], expected)
                    self.assertEqual(records[1]["correct"], 0)
                    self.assertEqual(len(records[1]["events"]), 2)
                    self.assertEqual(records[0]["reason"], "invalid_schema")
                    if matching == "integer_search":
                        self.assertEqual(metrics["answer_search_rate"], 0.5)
                        self.assertNotIn("pass_at_1", metrics)

    def test_single_agent_uses_the_same_configured_answer_matching(self) -> None:
        """Compare both scoring modes without running a model or initializing JAX."""
        with tempfile.TemporaryDirectory() as temporary:
            for matching, expected in (("strict", 0), ("integer_search", 1)):
                with self.subTest(matching=matching):
                    output = Path(temporary) / matching
                    output.mkdir()
                    settings = {"seed": 17, "maximum_messages": 8, "maximum_new_tokens": 256,
                                "finalization_reserve_seconds": 60, "answer_matching": matching}
                    study = Study(settings, {"deadline_unix_seconds": time.time() + 3600}, output)
                    study.backend = SimpleNamespace(prompt_width=1024, devices=[], batch_size=8,
                        generate_device=lambda *arguments: None,
                        generate_messages=lambda *arguments: [Generation((11,), (21,), (-0.3,), "The answer is 2.", True)])
                    with patch.dict("sys.modules", {"jax": SimpleNamespace(pmap=lambda *arguments, **keywords: None)}):
                        metrics = study.single_agent_baseline([Problem("fixture", "question", 2, "test")])
                    self.assertEqual(metrics["correct"], expected)
                    self.assertEqual(metrics["strict_correct"], 0)
                    if matching == "integer_search":
                        self.assertEqual(metrics["answer_search_rate"], 1)
                        self.assertNotIn("pass_at_1", metrics)

    def test_frozen_splits_preserve_order_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            files = {}
            for name in ("train", "validation", "test"):
                path = source / f"{name}.json"
                write_json(path, [{"identifier": f"{name}-{index}", "problem": f"{name} question {index}",
                                  "answer": index, "split": name} for index in (7, 2)])
                record = file_record(path)
                files[path.name] = {key: record[key] for key in ("bytes", "sha256")}
            settings = {"model": "fixture", "model_revision": "fixture", "maximum_prompt_tokens": 10,
                        "training_problem_limit": 2, "validation_problem_limit": 2, "evaluation_problem_limit": 2,
                        "frozen_data": {"directory": str(source), "region": "us-west4",
                                        "source_checkpoint": "gs://fixture/completed", "files": files}}
            output = root / "output"
            output.mkdir()
            tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                audit = prepare_data(settings, output, root / "cache", "us-west4")
                self.assertTrue(audit["reused_frozen_splits"])
                for name in ("train", "validation", "test"):
                    self.assertEqual((output / f"{name}.json").read_bytes(), (source / f"{name}.json").read_bytes())
                with self.assertRaisesRegex(ValueError, "original data region"):
                    prepare_data(settings, output, root / "cache", "us-central2")
                with self.assertRaisesRegex(ValueError, "prompt budget"):
                    prepare_data({**settings, "maximum_prompt_tokens": 2}, output, root / "cache", "us-west4")
                with self.assertRaisesRegex(ValueError, "split count"):
                    prepare_data({**settings, "training_problem_limit": 3}, output, root / "cache", "us-west4")
                (source / "train.json").write_text("corrupted")
                with self.assertRaisesRegex(ValueError, "integrity failed"):
                    prepare_data(settings, output, root / "cache", "us-west4")

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
            require_remote({**receipt, "deadline_unix_seconds": 8200}, Path("/home/kristinelu/peer-tree/source"), 7200)
            for deadline in (8201, 87400):
                with self.subTest(deadline=deadline), self.assertRaisesRegex(ValueError, "configured compute cap"):
                    require_remote({**receipt, "deadline_unix_seconds": deadline}, Path("/home/kristinelu/peer-tree/source"), 7200)


if __name__ == "__main__":
    unittest.main()
