import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIRECTORY = REPOSITORY_ROOT / ".agents" / "skills" / "query-monitor-gcp-tpus"
SWEEP_SCRIPT = SKILL_DIRECTORY / "scripts" / "sweep_tpus.sh"
OCCUPANCY_SCRIPT = SKILL_DIRECTORY / "scripts" / "check_tpu_occupancy.sh"
QUERY_SCRIPT = REPOSITORY_ROOT / "tpu_scripts" / "query_tpus.sh"
INITIALIZER_SCRIPT = REPOSITORY_ROOT / "tpu_scripts" / "initialize_elf_runtime.sh"
FOREIGN_HOLDER_GUARD_SCRIPT = (
    REPOSITORY_ROOT / "tpu_scripts" / "guard_relaunching_foreign_vfio_holder.sh"
)


class QueryMonitorTpuScriptsTest(unittest.TestCase):
    def test_foreign_holder_guard_tokenizes_fuser_output(self) -> None:
        """Handle repeated fuser PIDs even when its output has no final newline."""
        guard_source = FOREIGN_HOLDER_GUARD_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('fuser_output=', guard_source)
        self.assertIn('for process_identifier in \\${fuser_output}; do', guard_source)
        self.assertNotIn('done < <(sudo fuser', guard_source)
        self.assertIn('sort -nu', guard_source)

    def test_occupancy_check_supports_internal_tpu_ips(self) -> None:
        """Allow exact-target occupancy checks over the proven internal route."""
        occupancy_source = OCCUPANCY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("ELF_TPU_USE_INTERNAL_IP", occupancy_source)
        self.assertIn("ELF_TPU_USE_INTERNAL_IP:-1", occupancy_source)
        self.assertIn("gcloud_ssh_flags+=(--internal-ip)", occupancy_source)

    def make_fake_environment(
        self, temporary_directory: Path, mode: str
    ) -> tuple[dict[str, str], Path]:
        """Create a fake gcloud executable and return its environment and log."""
        binary_directory = temporary_directory / "bin"
        binary_directory.mkdir()
        command_log = temporary_directory / "gcloud.log"
        fake_gcloud = binary_directory / "gcloud"
        fake_gcloud.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${FAKE_GCLOUD_LOG}"
arguments=$*
if [[ "${arguments}" == *"tpu-vm create"* && "${FAKE_GCLOUD_MODE}" == 'reservation' ]]; then
    printf 'Reservation not found\n' >&2
    exit 1
elif [[ "${arguments}" == *"tpu-vm create"* && "${FAKE_GCLOUD_MODE}" == 'tenant' ]]; then
    printf 'Failed to perform tenant project creation\n' >&2
    exit 1
elif [[ "${arguments}" == *"asset search-all-resources"* ]]; then
    case "${FAKE_GCLOUD_MODE}" in
        sweep) printf '%s\\n' \
            'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/ready-kristine' \
            'us-central1-b,//tpu.googleapis.com/projects/p/locations/us-central1-b/nodes/pending-kristine' ;;
        unauthorized) printf '%s\\n' \
            'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/alice-tpu' ;;
    esac
elif [[ "${arguments}" == *"tpu-vm describe"* ]]; then
    if [[ "${arguments}" == *"pending-kristine"* ]]; then
        printf 'CREATING\\n'
    else
        printf 'READY\\n'
    fi
elif [[ "${arguments}" == *"tpu-vm ssh"* ]]; then
    if [[ "${arguments}" == *"sudo -n kill -TERM"* ]]; then
        printf 'CLAIMED worker=worker-0\\n'
    elif [[ "${FAKE_GCLOUD_MODE}" == 'occupied' ]]; then
        printf 'PROTECTED worker=worker-0 pid=42 user=kristinelu\\n'
        printf 'OCCUPIED worker=worker-0\\n'
    elif [[ "${FAKE_GCLOUD_MODE}" == 'reclaimable' ]]; then
        printf 'FOREIGN worker=worker-0 pid=42 user=other\\n'
        printf 'RECLAIMABLE worker=worker-0\\n'
    else
        printf 'AVAILABLE worker=worker-0\\n'
    fi
fi
""",
            encoding="utf-8",
        )
        fake_gcloud.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{binary_directory}:{environment['PATH']}",
                "FAKE_GCLOUD_LOG": str(command_log),
                "FAKE_GCLOUD_MODE": mode,
            }
        )
        return environment, command_log

    def run_script(
        self, script_path: Path, arguments: list[str], environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        """Run a skill script against the fake gcloud executable."""
        return subprocess.run(
            ["bash", str(script_path), *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_sweep_returns_only_current_ready_kristine_identifiers(self) -> None:
        """Return only exact Kristine IDs whose current API state is READY."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "sweep"
            )
            result = self.run_script(SWEEP_SCRIPT, ["project"], environment)
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ready-kristine\tus-central1-a\n")
        self.assertIn("--query=name:kristine", commands)
        self.assertIn("describe ready-kristine", commands)
        self.assertIn("describe pending-kristine", commands)

    def test_sweep_rejects_an_unexpected_non_kristine_result(self) -> None:
        """Stop before describing any non-Kristine result from asset inventory."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "unauthorized"
            )
            result = self.run_script(SWEEP_SCRIPT, ["project"], environment)
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertIn("unauthorized TPU ID", result.stderr)
        self.assertNotIn("tpu-vm describe", commands)

    def test_occupancy_rejects_non_kristine_id_before_gcloud(self) -> None:
        """Reject an unauthorized TPU ID before making any cloud call."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "available"
            )
            result = self.run_script(
                OCCUPANCY_SCRIPT,
                ["project", "alice-tpu", "us-central1-a"],
                environment,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing non-kristine", result.stderr)
        self.assertFalse(command_log.exists())

    def test_initializer_rejects_non_kristine_id_without_exact_authorization(
        self,
    ) -> None:
        """Keep runtime initialization restricted to owned or exactly authorized TPUs."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            missing_netrc_path = Path(temporary_directory_name) / ".netrc"
            environment = os.environ.copy()
            environment["ELF_NETRC_PATH"] = str(missing_netrc_path)
            environment.pop("ELF_TPU_AUTHORIZED_NON_KRISTINE_ID", None)
            result = self.run_script(
                INITIALIZER_SCRIPT,
                ["project", "alice-tpu", "us-central1-a"],
                environment,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing non-kristine TPU ID: alice-tpu", result.stderr)

    def test_initializer_requires_the_exact_authorized_non_kristine_id(self) -> None:
        """Reject a non-owned TPU when authorization names a different resource."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            missing_netrc_path = Path(temporary_directory_name) / ".netrc"
            environment = os.environ.copy()
            environment.update(
                {
                    "ELF_NETRC_PATH": str(missing_netrc_path),
                    "ELF_TPU_AUTHORIZED_NON_KRISTINE_ID": "other-tpu",
                }
            )
            result = self.run_script(
                INITIALIZER_SCRIPT,
                ["project", "alice-tpu", "us-central1-a"],
                environment,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing non-kristine TPU ID: alice-tpu", result.stderr)

    def test_initializer_accepts_the_exact_authorized_non_kristine_id(self) -> None:
        """Allow the exact acquired TPU to advance to credential validation."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            missing_netrc_path = Path(temporary_directory_name) / ".netrc"
            environment = os.environ.copy()
            environment.update(
                {
                    "ELF_NETRC_PATH": str(missing_netrc_path),
                    "ELF_TPU_AUTHORIZED_NON_KRISTINE_ID": "alice-tpu",
                }
            )
            result = self.run_script(
                INITIALIZER_SCRIPT,
                ["project", "alice-tpu", "us-central1-a"],
                environment,
            )

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("refusing non-kristine", result.stderr)
        self.assertIn("ELF W&B credential file is missing", result.stderr)

    def test_occupancy_checks_libtpu_lockfile_on_v4(self) -> None:
        """Detect held and holderless libtpu locks without mutating either one."""
        occupancy_source = OCCUPANCY_SCRIPT.read_text(encoding="utf-8")

        self.assertGreaterEqual(occupancy_source.count("/tmp/libtpu_lockfile"), 2)
        self.assertGreaterEqual(
            occupancy_source.count("reason=stale_libtpu_lock"), 2
        )
        self.assertGreaterEqual(
            occupancy_source.count("lockfile_holder_found"), 6
        )
        self.assertNotIn("rm -- /tmp/libtpu_lockfile", occupancy_source)

    def test_occupancy_ignores_the_host_ssh_configuration(self) -> None:
        """Keep occupancy inspection independent of a malformed host SSH config."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "available"
            )
            result = self.run_script(
                OCCUPANCY_SCRIPT,
                ["project", "ready-kristine", "us-central1-a"],
                environment,
            )
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ssh-flag=-F --ssh-flag=/dev/null", commands)

    def test_occupancy_protects_active_trainers_without_device_holders(self) -> None:
        """Never claim a v4 while a live trainer is invisible to device inspection."""
        occupancy_source = OCCUPANCY_SCRIPT.read_text(encoding="utf-8")

        self.assertGreaterEqual(occupancy_source.count("training_entrypoint=train.py"), 2)
        self.assertIn("reason=active_train_process", occupancy_source)
        self.assertIn("protected_holder_found=1", occupancy_source)

    def test_occupancy_ignores_only_the_validated_tpu_name_in_holder_metadata(self) -> None:
        """Do not classify foreign work as Kristine's solely by the TPU resource name."""
        occupancy_source = OCCUPANCY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("TPU ID contains unsupported characters", occupancy_source)
        resource_name_pattern = (
            "s/kmh-tpuvm-[A-Za-z0-9._-]*kristine[A-Za-z0-9._-]*//g"
        )
        self.assertGreaterEqual(occupancy_source.count(resource_name_pattern), 4)
        self.assertIn('searchable_process="${process_owner}', occupancy_source)

    def test_claim_does_not_touch_a_protected_holder(self) -> None:
        """Do not run the claim phase when any protected VFIO holder exists."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "occupied"
            )
            result = self.run_script(
                OCCUPANCY_SCRIPT,
                ["project", "ready-kristine", "us-central1-a", "--claim"],
                environment,
            )
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 2)
        self.assertIn("OCCUPIED", result.stdout)
        self.assertEqual(commands.count("tpu-vm ssh"), 1)

    def test_claim_runs_only_after_reclaimable_inspection(self) -> None:
        """Run the termination phase only after finding foreign VFIO holders."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "reclaimable"
            )
            result = self.run_script(
                OCCUPANCY_SCRIPT,
                ["project", "ready-kristine", "us-central1-a", "--claim"],
                environment,
            )
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECLAIMABLE", result.stdout)
        self.assertIn("CLAIMED", result.stdout)
        self.assertEqual(commands.count("tpu-vm ssh"), 2)

    def test_query_script_rejects_unsupported_topology(self) -> None:
        """Reject v6e sizes below the supported eight-chip query floor."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "available"
            )
            result = self.run_script(QUERY_SCRIPT, ["v6e", "4", "8"], environment)

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "v6e amount must be 8, 16, 32, 64, 128, or 256",
            result.stdout,
        )
        self.assertFalse(command_log.exists())

    def test_query_maps_large_v6e_topologies_to_supported_runtime(self) -> None:
        """Create every requested large v6e topology with the v6e runtime."""
        for chip_count in ("64", "128", "256"):
            with self.subTest(chip_count=chip_count):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    temporary_directory = Path(temporary_directory_name)
                    environment, command_log = self.make_fake_environment(
                        temporary_directory,
                        "available",
                    )
                    initializer_path = temporary_directory / "initialize.sh"
                    initializer_path.write_text(
                        "#!/bin/sh\nexit 0\n",
                        encoding="utf-8",
                    )
                    initializer_path.chmod(0o755)
                    for executable_name in ("curl", "sleep"):
                        fake_executable_path = (
                            temporary_directory / "bin" / executable_name
                        )
                        fake_executable_path.write_text(
                            "#!/bin/sh\nexit 0\n",
                            encoding="utf-8",
                        )
                        fake_executable_path.chmod(0o755)
                    environment.update(
                        {
                            "ELF_TPU_INITIALIZER": str(initializer_path),
                            "ZONES_OVERRIDE": "us-east5-b",
                        }
                    )

                    result = self.run_script(
                        QUERY_SCRIPT,
                        ["v6e", chip_count, "1"],
                        environment,
                    )
                    create_command = next(
                        command
                        for command in command_log.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if "tpu-vm create" in command
                    )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"--accelerator-type=v6e-{chip_count}",
                    create_command,
                )
                self.assertIn("--version=v2-alpha-tpuv6e", create_command)
                self.assertIn(
                    f"machine_type=v6e-{chip_count} "
                    f"machines_per_tpu={int(chip_count) // 4} "
                    f"total_machines={int(chip_count) // 4}",
                    result.stdout,
                )

    def test_query_rejects_family_zone_mismatches_before_cloud_access(self) -> None:
        """Fail before creation when an override is invalid for the TPU family."""
        invalid_combinations = (
            ("v5e", "64", "us-central2-b"),
            ("v5p", "32", "asia-northeast1-b"),
            ("v6e", "64", "us-west4-a"),
            ("v4", "32", "us-east5-b"),
        )
        for accelerator_type, chip_count, zone in invalid_combinations:
            with self.subTest(accelerator_type=accelerator_type, zone=zone):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    environment, command_log = self.make_fake_environment(
                        Path(temporary_directory_name),
                        "available",
                    )
                    environment["ZONES_OVERRIDE"] = zone
                    result = self.run_script(
                        QUERY_SCRIPT,
                        [accelerator_type, chip_count, "1"],
                        environment,
                    )

                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    f"zone {zone} does not support {accelerator_type}",
                    result.stderr,
                )
                self.assertFalse(command_log.exists())

    def test_query_rejects_an_empty_zone_override_before_cloud_access(self) -> None:
        """Reject a whitespace-only zone override instead of entering an empty loop."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name),
                "available",
            )
            environment["ZONES_OVERRIDE"] = "   "
            result = self.run_script(
                QUERY_SCRIPT,
                ["v6e", "64", "1"],
                environment,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "ZONES_OVERRIDE must contain at least one zone",
            result.stderr,
        )
        self.assertFalse(command_log.exists())

    def test_query_maps_v5e_64_to_v5litepod_runtime(self) -> None:
        """Create v5e resources with Cloud TPU's v5litepod type and runtime names."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-a",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "64", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--accelerator-type=v5litepod-64", create_command)
        self.assertIn("--version=v2-alpha-tpuv5-lite", create_command)

    def test_query_maps_v5e_4_to_one_v5litepod_vm(self) -> None:
        """Represent the four-chip v5e slice as one TPU VM."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-a",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "4", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--accelerator-type=v5litepod-4", create_command)
        self.assertIn("machine_type=v5litepod-4 machines_per_tpu=1 total_machines=1", result.stdout)

    def test_query_uses_us_east5_bucket_and_elf_initializer_contract(self) -> None:
        """Create only an owned us-east5 TPU with the region-bound ELF identity."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_log = temporary_directory / "initializer.log"
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$INITIALIZER_LOG\"\n",
                encoding="utf-8",
            )
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "INITIALIZER_LOG": str(initializer_log),
                    "ZONES_OVERRIDE": "us-east5-b",
                }
            )

            result = self.run_script(
                QUERY_SCRIPT,
                ["v6e", "8", "1"],
                environment,
            )
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )
            initializer_arguments = initializer_log.read_text(
                encoding="utf-8"
            ).strip().split()
            query_source = QUERY_SCRIPT.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--zone=us-east5-b", create_command)
        self.assertIn(
            "--service-account="
            "bucket-us-east5@he-vision-group.iam.gserviceaccount.com",
            create_command,
        )
        self.assertRegex(
            create_command,
            r"tpu-vm create kmh-tpuvm-v6e-8-spot-kristine-.*-us-east5-b ",
        )
        self.assertEqual(
            initializer_arguments[0], "he-vision-group"
        )
        self.assertRegex(
            initializer_arguments[1],
            r"^kmh-tpuvm-v6e-8-spot-kristine-.*-us-east5-b$",
        )
        self.assertEqual(initializer_arguments[2], "us-east5-b")
        self.assertIn(
            "v6e) printf '%s\\n'",
            query_source,
        )
        self.assertIn(
            "'us-central1-b us-central1-a us-east5-a asia-northeast1-b "
            "us-east5-b us-central1-c'",
            query_source,
        )
        self.assertIn(
            "initializer=${ELF_TPU_INITIALIZER:-"
            "${script_directory}/initialize_elf_runtime.sh}",
            query_source,
        )
        self.assertNotIn("run_init_remote.sh", query_source)

    def test_query_allows_v5e_8_in_pack_backed_us_east5(self) -> None:
        """Use the surveyed v5litepod type with the matching east5 identity."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-east5-b",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--accelerator-type=v5litepod-8", create_command)
        self.assertIn("--zone=us-east5-b", create_command)
        self.assertIn(
            "--service-account="
            "bucket-us-east5@he-vision-group.iam.gserviceaccount.com",
            create_command,
        )

    def test_query_stops_retrying_a_non_retryable_reservation_blocker(self) -> None:
        """Fail cleanly instead of looping forever on a reservation-only offering."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "reservation",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-east5-b",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_commands = [
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            ]

        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(create_commands), 1)
        self.assertIn("Non-retryable create blocker", result.stderr)
        self.assertIn("none:none:BLOCKED", result.stdout)

    def test_query_stops_retrying_a_tenant_project_blocker(self) -> None:
        """Treat a repeated TPU tenant-project failure as deterministic."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "tenant",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-b",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_commands = [
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            ]

        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(create_commands), 1)
        self.assertIn("Non-retryable create blocker", result.stderr)
        self.assertIn("none:none:BLOCKED", result.stdout)

    def test_query_uses_us_west4_bucket_service_account(self) -> None:
        """Create us-west4 TPUs with the matching regional bucket identity."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-a",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--zone=us-west4-a", create_command)
        self.assertIn(
            "--service-account="
            "bucket-us-west4@he-vision-group.iam.gserviceaccount.com",
            create_command,
        )


if __name__ == "__main__":
    unittest.main()
