"""Check that audit authorization and source integrity fail before experiment imports."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from audit_checkpoints import verify_execution_receipt, verify_source_manifest


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Path("/home/kristinelu/smollm2-gsm8k/matched-validation-20261103/source")
        self.hardware = {"project": "he-vision-group", "zone": "us-west4-a",
            "name": "kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a",
            "type": "v5litepod-8"}
        self.protocol = {"study_id": "unit-test-only", "hardware": self.hardware,
                         "budget": {"finalization_reserve_seconds": 3600}}
        self.receipt = {"approval_status": "approved", "study_id": "unit-test-only",
            "hardware": deepcopy(self.hardware), "maximum_concurrent_tpus": 1,
            "verified_hostname": "unit-test-host", "spot": True, "development_machine": False,
            "study_started_unix_seconds": 1000.0, "authorized_seconds": 14400}

    def test_matching_receipt_preserves_original_budget_start(self) -> None:
        budget = verify_execution_receipt(self.receipt, self.protocol, self.source, "unit-test-host", 2000.0)
        self.assertEqual(budget.started_unix_seconds, 1000.0)
        self.assertEqual(budget.remaining(2000.0), 13400)

    def test_pending_budget_does_not_authorize_model_imports(self) -> None:
        self.receipt["approval_status"] = "pending"
        with self.assertRaises(PermissionError):
            verify_execution_receipt(self.receipt, self.protocol, self.source, "unit-test-host", 2000.0)

    def test_dev_host_wrong_path_or_expanded_capacity_fails(self) -> None:
        with self.assertRaises(PermissionError):
            verify_execution_receipt(self.receipt, self.protocol, self.source, "dev-host", 2000.0)
        with self.assertRaises(PermissionError):
            verify_execution_receipt(self.receipt, self.protocol, Path("/tmp/source"), "unit-test-host", 2000.0)
        for field, value in [("development_machine", True), ("maximum_concurrent_tpus", 2), ("spot", False)]:
            receipt = {**self.receipt, field: value}
            with self.assertRaises(PermissionError):
                verify_execution_receipt(receipt, self.protocol, self.source, "unit-test-host", 2000.0)

    def test_other_tpu_or_expired_window_fails(self) -> None:
        self.receipt["hardware"]["name"] = "some-other-tpu"
        with self.assertRaises(PermissionError):
            verify_execution_receipt(self.receipt, self.protocol, self.source, "unit-test-host", 2000.0)
        self.receipt["hardware"] = deepcopy(self.hardware)
        with self.assertRaises(TimeoutError):
            verify_execution_receipt(self.receipt, self.protocol, self.source, "unit-test-host", 12000.0)

    def test_modified_source_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "example.py"
            source.write_text("value = 1\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = directory / "source.sha256"
            manifest.write_text(digest + "  example.py\n")
            self.assertEqual(verify_source_manifest(directory), hashlib.sha256(manifest.read_bytes()).hexdigest())
            source.write_text("value = 2\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_source_manifest(directory)

    def test_manifest_cannot_escape_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "source.sha256").write_text("0" * 64 + "  ../other.py\n")
            with self.assertRaises(ValueError):
                verify_source_manifest(directory)


if __name__ == "__main__":
    unittest.main()
