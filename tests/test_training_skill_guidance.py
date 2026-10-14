from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SKILL_PATH = (
    REPOSITORY_ROOT / ".agents" / "skills" / "launch-elf-training" / "SKILL.md"
)
ERRORS_SKILL_PATH = REPOSITORY_ROOT / ".agents" / "skills" / "errors" / "SKILL.md"
OPERATIONAL_LESSONS_PATH = (
    REPOSITORY_ROOT
    / ".agents"
    / "skills"
    / "launch-elf-training"
    / "references"
    / "operational_lessons.md"
)
QUERY_MONITOR_SKILL_PATH = (
    REPOSITORY_ROOT / ".agents" / "skills" / "query-monitor-gcp-tpus" / "SKILL.md"
)
REPOSITORY_AGENT_GUIDANCE_PATH = REPOSITORY_ROOT / "AGENTS.md"


def test_launch_skill_gates_linear_masks_and_effective_examples_per_second() -> None:
    """Require the launch workflow to reject quadratic host masks and slow smokes."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")

    assert "only linear `[batch, sequence]` masks" in launch_guidance
    assert "never materialize or transport a quadratic" in launch_guidance
    assert "inside the compiled train or evaluation step on the TPU" in launch_guidance
    assert "training_loop_steps_in_interval * global_batch_size" in launch_guidance
    assert "Never multiply every loop step by `grad_accum_steps`" in launch_guidance
    assert "effective_batch_size = global_batch_size * grad_accum_steps" in launch_guidance
    assert "denominator must span that full accumulation window" in launch_guidance
    assert "several consecutive steady-state steps" in launch_guidance
    assert "accepted benchmark" in launch_guidance
    assert "block full promotion" in launch_guidance
    assert "On confirmed TPU loss, keep one recovery orchestrator" in launch_guidance
    assert "CLOUDSDK_PYTHON" in launch_guidance
    assert "restart exactly one mirror" in launch_guidance
    assert "synchronous dataloader replay" in launch_guidance
    assert "deterministic distributed-sampler batch offset" in launch_guidance
    assert "Never launch a duplicate recovery from W&B's `crashed` label alone" in launch_guidance

    operational_lessons = " ".join(
        OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8").split()
    )
    assert "project decoder vocabulary logits only for supervised suffix positions" in operational_lessons
    assert "Never infer a fixed prefix from a maximum length" in operational_lessons


def test_launch_skill_commits_only_the_selected_run_config_closure() -> None:
    """Keep launch PRs from accumulating unused YAML candidates."""
    launch_guidance = " ".join(
        LAUNCH_SKILL_PATH.read_text(encoding="utf-8").split()
    )

    assert "smallest reproducible configuration closure" in launch_guidance
    assert "Normally commit one primary run YAML" in launch_guidance
    assert "exact file passed to the launcher" in launch_guidance
    assert "Never bulk-stage a configuration directory" in launch_guidance
    assert "sweep all modified and untracked YAMLs" in launch_guidance
    assert "unrelated, abandoned, superseded" in launch_guidance
    assert "immediate launch under its own run identity" in launch_guidance
    assert "only when that run is launched and monitored" in launch_guidance
    assert "git diff --cached --name-only" in launch_guidance
    assert "primary config path, commit SHA, pull request URL" in launch_guidance


def test_errors_skill_catalogs_quadratic_mask_regression_without_losing_recovery() -> None:
    """Keep the throughput postmortem and existing Spot recovery rules together."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")

    assert "### Host-side quadratic-mask throughput regression" in error_guidance
    assert "linear `attention_mask` and `cond_seq_mask`" in error_guidance
    assert "training_loop_steps * global_batch_size" in error_guidance
    assert "never multiply each loop step by it" in error_guidance
    assert "fail the throughput gate, block full promotion" in error_guidance
    assert "Spot preemption or TPU deletion" in error_guidance
    assert "use equivalent authorized `kristine` Spot replacement" in error_guidance
    assert "TypeError: 'type' object is not subscriptable" in error_guidance
    assert "start exactly one replacement helper" in error_guidance


def test_skills_recover_when_legacy_gcloud_has_cp_but_not_rsync() -> None:
    """Keep exact local dataset reuse independent of a broken gsutil fallback."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    for guidance in (error_guidance, operational_lessons):
        assert "gcloud storage ls" in guidance
        assert "gcloud storage rsync" in guidance
        assert "gcloud storage cp" in guidance
        assert "exact bytes" in guidance
        assert "temporary files" in guidance
        assert "SDK `428.0.0`" in guidance
    assert "must not require a successful `gsutil du`" in error_guidance
    assert "Check exact local reuse before invoking any" in operational_lessons


def test_launch_skill_reconciles_successors_before_preemption_recovery() -> None:
    """Require full-lineage reconciliation before any recovery side effect."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")

    assert "Before any preemption recovery, capacity query, or checkpoint localization" in launch_guidance
    assert "all experiment reports" in launch_guidance
    assert "every active installed monitor and recovery orchestrator" in launch_guidance
    assert "authenticated read-only W&B state" in launch_guidance
    assert "If a newer active successor exists" in launch_guidance
    assert "Retire only the obsolete predecessor's monitor and recovery trigger" in launch_guidance


def test_launch_skill_allows_guarded_v5p_fallback_after_v6e_loss() -> None:
    """Keep v5p recovery available without weakening locality or compatibility gates."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")

    assert "v5p-8, v5p-16, or v5p-32 is an approved fallback" in launch_guidance
    assert "checkpoint, dataset, and future outputs remain in its region" in launch_guidance
    assert "never silently change model hyperparameters" in launch_guidance
    assert "Never create a duplicate full run from an older checkpoint" in launch_guidance
    assert "On every recovery, re-push that snapshot and selected configuration" in launch_guidance
    assert "never rely on files from a prior TPU" in launch_guidance


def test_launch_skill_requires_optimal_tpu_sweep_and_guarded_checkpoint_copy() -> None:
    """Require launch requests to choose capacity before permitting one checkpoint copy."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")

    assert "Whenever the user asks to launch, resume, relocate, or update a run" in launch_guidance
    assert "check_tpus.sh he-vision-group" in launch_guidance
    assert "never choose solely by accelerator count" in launch_guidance
    assert "A single finalized checkpoint may be copied once between regional buckets" in launch_guidance
    assert "This exception never applies to datasets" in launch_guidance
    assert "Never copy a dataset across regions" in launch_guidance
    assert "First inspect the selected TPU region's `kristine` bucket path" in launch_guidance
    assert "only when that matching regional `kristine` path is absent" in launch_guidance
    assert "If disk capacity is insufficient, first prove that the selected run's complete verified dataset is absent" in launch_guidance
    assert "never remove checkpoints, outputs, source snapshots, credentials" in launch_guidance


def test_launch_guidance_requires_explicit_tpu_provisioning_request() -> None:
    """Keep ordinary launch and recovery requests on existing inventoried TPUs."""
    guidance_documents = (
        LAUNCH_SKILL_PATH.read_text(encoding="utf-8"),
        OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8"),
        QUERY_MONITOR_SKILL_PATH.read_text(encoding="utf-8"),
        REPOSITORY_AGENT_GUIDANCE_PATH.read_text(encoding="utf-8"),
    )

    for guidance_document in guidance_documents:
        normalized_guidance = " ".join(guidance_document.split())
        assert "check_tpus.sh" in normalized_guidance
        assert "explicitly requests new TPU provisioning" in normalized_guidance
        assert "query_tpus.sh" in normalized_guidance


def test_tpu_workflow_keeps_non_kristine_sweeps_read_only() -> None:
    """Allow other TPUs only through the explicit read-only and idle-acquisition phrases."""
    guidance_documents = (
        LAUNCH_SKILL_PATH.read_text(encoding="utf-8"),
        ERRORS_SKILL_PATH.read_text(encoding="utf-8"),
        OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8"),
        QUERY_MONITOR_SKILL_PATH.read_text(encoding="utf-8"),
        REPOSITORY_AGENT_GUIDANCE_PATH.read_text(encoding="utf-8"),
    )

    for guidance_document in guidance_documents:
        normalized_guidance = " ".join(guidance_document.split())
        assert "AUTHORIZE WIDE READ-ONLY TPU SWEEP" in normalized_guidance
        assert "AUTHORIZE IDLE NON-KRISTINE TPU ACQUISITION" in normalized_guidance
        assert "make yourself known to a run" in normalized_guidance
        assert "A later full sweep may read-only re-query a TPU previously found actively held, because its job may have finished and the TPU may be inactive again." in normalized_guidance
        assert "Every sweep stays read-only and must never touch an existing active job." in normalized_guidance
        assert "On a non-`kristine` TPU, do not edit, stop, or touch an actively held run owned by somebody else at any point, even when the user requests it." in normalized_guidance
        assert "A process owned by kristine may be terminated only when the user explicitly authorizes that termination, and never as part of a sweep." in normalized_guidance
        assert "If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it." in normalized_guidance
        assert "On a `kristine` TPU, the documented recovery rules apply." in normalized_guidance

    query_guidance = " ".join(QUERY_MONITOR_SKILL_PATH.read_text(encoding="utf-8").split())
    assert "Never use `--claim` during a wide sweep" in query_guidance
    assert "never use it on a TPU whose name lacks `kristine`" in query_guidance


def test_errors_skill_catalogs_duplicate_successor_recovery() -> None:
    """Keep obsolete predecessor monitors from spawning duplicate full runs."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")

    assert "### Obsolete precursor recovery would duplicate an active successor" in error_guidance
    assert "all experiment reports, active installed monitors/recovery orchestrators" in error_guidance
    assert "authenticated read-only W&B lineage" in error_guidance
    assert "retire only the obsolete predecessor monitor and recovery trigger" in error_guidance
    assert "Never create a duplicate full run from an older checkpoint" in error_guidance


def test_launch_guidance_keeps_the_final_occupancy_gate_read_only() -> None:
    """Prevent the final launch-race gate from rerunning mutating setup logic."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    for guidance in (error_guidance, operational_lessons):
        assert "dedicated read-only occupancy check" in guidance
        assert "never reuse a setup preflight" in guidance


def test_runtime_signature_guidance_normalizes_only_transport_whitespace() -> None:
    """Keep SSH formatting from creating a false mismatch without weakening payloads."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    assert "### Identical runtime signatures fail comparison after SSH fan-out" in error_guidance
    assert "both CRLF and space-padded duplicates" in error_guidance
    assert "canonicalizing SSH transport" in operational_lessons
    assert "genuinely different payloads must still fail closed" in operational_lessons


def test_skills_separate_training_and_cloud_sdk_python_compatibility() -> None:
    """Require an exact-worker Cloud SDK probe instead of inheriting training Python."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")

    for guidance in (launch_guidance, error_guidance):
        assert "separate compatibility domains" in guidance
        assert "Never blindly set `CLOUDSDK_PYTHON` to `ELF_RUNTIME_PYTHON`" in guidance
        assert "exact `gcloud storage` command shape" in guidance
        assert "Cloud SDK 127" in guidance
        assert "Python 3.12" in guidance
        assert "`imp`" in guidance
        assert "system Python 3.8" in guidance
        assert "ELF Python 3.11" in guidance
        assert "bundled SDK default" in guidance
        assert "fail closed" in guidance.lower()


def test_skills_bound_fresh_checkpoint_mirror_grace() -> None:
    """Prevent cadence-aligned upload windows from becoming false recovery events."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")

    assert "ELF_MONITOR_CHECKPOINT_MIRROR_GRACE_SECONDS" in launch_guidance
    assert "`pending_grace` without incrementing infrastructure failures" in launch_guidance
    assert "older than the grace" in launch_guidance
    assert "### Checkpoint cadence aliases with monitor cadence" in error_guidance
    assert "false consecutive-failure threshold" in error_guidance
    assert "Do not trigger TPU recovery from a fresh unmirrored checkpoint alone" in error_guidance
    assert "After the grace expires" in error_guidance


def test_skills_treat_verified_terminal_zombies_as_completion() -> None:
    """Prevent normal trainer teardown from launching a duplicate retry."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")
    normalized_error_guidance = " ".join(error_guidance.split())
    normalized_operational_lessons = " ".join(operational_lessons.split())

    assert "### Terminal trainer and mirror children remain as zombies" in error_guidance
    assert "exact `train.py` argv token" in error_guidance
    assert "verified terminal checkpoint" in error_guidance
    assert "never launch a retry" in normalized_error_guidance
    assert "live non-zombie `/proc` entry" in normalized_operational_lessons
    assert "W&B is `finished`" in normalized_operational_lessons
    assert "terminal checkpoint marker" in normalized_operational_lessons


def test_skills_close_monitor_lock_in_persistent_recovery_children() -> None:
    """Keep recurring checks active while one Codex recovery process runs."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    assert "### Recovery child inherits the monitor flock" in error_guidance
    assert "close the flock descriptor" in error_guidance
    assert "`9>&-`" in error_guidance
    assert "persistent fake recovery process" in error_guidance
    assert "close its flock file" in operational_lessons
    assert "sole orchestrator" in operational_lessons


def test_skills_isolate_successor_monitors_from_precursor_cgroups() -> None:
    """Prevent precursor retirement from killing its authoritative successor."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    assert "### Successor monitor inherits an obsolete precursor systemd cgroup" in error_guidance
    assert "inspect `/proc/<pid>/cgroup`" in error_guidance
    assert "dedicated unit" in error_guidance
    assert "cleanup helper in a third cgroup" in error_guidance
    assert "Never signal the remote trainer" in error_guidance
    assert "Shell detachment is also not systemd-cgroup isolation" in operational_lessons
    assert "Never stop the precursor" in operational_lessons
    assert "exactly one cadence-sleep child" in operational_lessons


def test_skills_monitor_log_and_mirror_workers_independently() -> None:
    """Keep automatic JAX process zero from hiding a healthy mirror helper."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")
    normalized_error_guidance = " ".join(error_guidance.split())

    assert "### Mirror helper is checked on the process-zero log worker" in error_guidance
    assert "independent non-negative log-worker and mirror-worker indices" in normalized_error_guidance
    assert "Do not move or duplicate the helper" in error_guidance
    assert "automatic JAX process zero" in operational_lessons
    assert "Pin them independently" in operational_lessons
    assert "Keep that helper running" in operational_lessons


def test_skills_use_process_local_replicas_for_multihost_checkpoints() -> None:
    """Prevent JAX global-array indexing from failing at a checkpoint boundary."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")
    normalized_operational_lessons = " ".join(operational_lessons.split())

    assert "### Globally sharded pmap state is indexed" in error_guidance
    assert "`tree_map(lambda x: x[0], state)`" in error_guidance
    assert "completion-marker, byte-count, SHA-256" in error_guidance
    assert "array.addressable_shards[0].data.squeeze(0)" in error_guidance
    assert "Read one addressable replica when saving JAX 0.11 pmap state" in normalized_operational_lessons
    assert "rejects `jax.Array.__getitem__`" in normalized_operational_lessons
    assert "payload-step verification" in normalized_operational_lessons


def test_launch_skill_requires_the_accumulated_operational_runbook() -> None:
    """Keep durable launch and evaluation lessons attached to every invocation."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")
    normalized_operational_lessons = " ".join(operational_lessons.split())

    assert "read" in launch_guidance.lower()
    assert "references/operational_lessons.md" in launch_guidance
    assert "for every ELF TPU launch" in normalized_operational_lessons
    assert "use only its final `AVAILABLE` table" in normalized_operational_lessons
    assert "exact-name describe and occupancy calls" in normalized_operational_lessons
    assert "highest exact common checkpoint only when both files still exist" in normalized_operational_lessons
    assert "never infer availability" in normalized_operational_lessons
    assert "USE_TORCH_XLA=0" in normalized_operational_lessons
    assert "`-o GSUtil:check_hashes=never`" in normalized_operational_lessons
    assert "independently trusted byte count and SHA-256" in normalized_operational_lessons
    assert "17.56 MiB" in normalized_operational_lessons
    assert "identical batch seeds, initial noise, timesteps" in normalized_operational_lessons
    assert "Cache immutable reference features" in normalized_operational_lessons
    assert "reference-content digest, model revision" in normalized_operational_lessons
    assert "Never change a running evaluation mid-arm" in normalized_operational_lessons
    assert "ELF_MONITOR_EVAL_INTERVAL_STEPS=0" in normalized_operational_lessons
    assert "not a scientific completion validator" in normalized_operational_lessons
    assert "process-exit/W&B-state race as pending" in normalized_operational_lessons
    assert "output/config.yml" in normalized_operational_lessons


def test_operational_runbook_initializes_only_explicitly_missing_runtimes() -> None:
    """Initialize during staging without turning transport errors into mutations."""
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")
    normalized_operational_lessons = " ".join(operational_lessons.split())

    assert "stage_elf_to_tpu.sh" in normalized_operational_lessons
    assert "explicitly reports `ELF_RUNTIME_MISSING`" in normalized_operational_lessons
    assert "malformed probe is unknown" in normalized_operational_lessons
    assert "fail without initialization" in normalized_operational_lessons
    assert "ELF_TPU_INITIALIZE_MISSING=0" in normalized_operational_lessons
    assert "rerun the complete readiness probe" in normalized_operational_lessons


def test_guidance_blocks_jax_logging_before_distributed_initialization() -> None:
    """Keep retry metadata logging from opening the JAX backend too early."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    assert "### Pre-distributed logging initializes the JAX backend" in error_guidance
    assert "`log_for_0()`, which calls `jax.process_index()`" in error_guidance
    assert "before the first `jax.process_index()`" in error_guidance
    assert "wandb_run_id=None" in error_guidance
    assert "never call `log_for_0()`" in operational_lessons
    assert "launcher-provided worker index" in operational_lessons
    assert "literal string `None`" in operational_lessons


def test_guidance_keeps_wandb_and_post_jax_telemetry_on_one_worker() -> None:
    """Prevent physical W&B ownership from diverging from metric ownership."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    for guidance in (error_guidance, operational_lessons):
        normalized_guidance = " ".join(guidance.split())
        assert "physical worker zero" in normalized_guidance
        assert "W&B metrics" in normalized_guidance
        assert "scientific" in normalized_guidance
        assert "JAX process" in normalized_guidance


def test_operational_runbook_handles_exact_evaluation_terminal_states_safely() -> None:
    """Treat scientific collapse as completion without enabling generic recovery."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")
    normalized_operational_lessons = " ".join(operational_lessons.split())

    assert "exact-checkpoint generation evaluations" in launch_guidance
    assert "exact restored checkpoint plus its first useful evaluation progress event" in launch_guidance
    assert "all-empty or whitespace-only arm is a completed scientific collapse" in normalized_operational_lessons
    assert "skipped PPL or MAUVE as undefined" in normalized_operational_lessons
    assert "ELF_MONITOR_TRIGGER_CODEX=false" in normalized_operational_lessons
    assert "ELF_MONITOR_IGNORE_TERMINAL_WANDB_WHILE_TRAINING=true" in normalized_operational_lessons
    assert "wandb_run_id: null" in normalized_operational_lessons
    assert "W&B assigns the run ID" in normalized_operational_lessons
    assert "bounded by the next direct orchestrator check" in normalized_operational_lessons
    assert "inside one W&B run, not separate W&B runs" in normalized_operational_lessons


def test_operational_runbook_preserves_localized_data_and_paired_retry_integrity() -> None:
    """Keep sequential exact evaluations fast and prevent partial-arm reuse."""
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")
    normalized_operational_lessons = " ".join(operational_lessons.split())

    assert "every expected final Arrow filename" in normalized_operational_lessons
    assert "zero `*.gstmp` transfer remnants" in normalized_operational_lessons
    assert "`cache-*.arrow` files may exist outside the canonical source count" in normalized_operational_lessons
    assert "never start an overlapping retry while it remains live" in normalized_operational_lessons
    assert "same destination filesystem" in normalized_operational_lessons
    assert "digest must predate and be independent" in normalized_operational_lessons
    assert "two direct checks at least 60 seconds apart" in normalized_operational_lessons
    assert "rerun the complete pair from the same exact verified checkpoint" in normalized_operational_lessons
    assert "Never append a treatment arm" in normalized_operational_lessons
    assert "precursor's byte-identical source" in normalized_operational_lessons
    assert "may not substitute another step or restart from step zero" in normalized_operational_lessons
    assert "remove only that run's cron entry" in normalized_operational_lessons
    assert "preserve every unrelated installed monitor" in normalized_operational_lessons
    assert "future two-phase path" in normalized_operational_lessons
    assert "Diff every parameter-bearing and source-preprocessing field" in normalized_operational_lessons
    assert "if any material field" in normalized_operational_lessons
    assert "An exact-checkpoint evaluation must stop" in LAUNCH_SKILL_PATH.read_text(encoding="utf-8")


def test_exact_evaluations_share_the_source_training_wandb_group() -> None:
    """Group checkpoint evaluations by verified source-run ancestry."""
    launch_guidance = LAUNCH_SKILL_PATH.read_text(encoding="utf-8")
    operational_lessons = OPERATIONAL_LESSONS_PATH.read_text(encoding="utf-8")

    for guidance in (launch_guidance, operational_lessons):
        assert "set `wandb_group` to that exact source run ID" in guidance
        assert "checkpoint ancestry" in guidance
        assert "sampling policies" in guidance
        assert "W&B run ID" in guidance
        assert "explicit user request" in guidance


def test_error_guidance_rejects_escaped_multiline_remote_commands() -> None:
    """Preserve zero-step failures caused by remote command transport quoting."""
    error_guidance = ERRORS_SKILL_PATH.read_text(encoding="utf-8")

    assert "### Multiline remote command is flattened into escaped text" in error_guidance
    assert "shell-quoted single-line command" in error_guidance
    assert "verify every exact target and" in error_guidance
    assert "temporary path is absent" in error_guidance
    assert "local shell, gcloud, SSH, and remote shell boundaries" in error_guidance
