#!/usr/bin/env bash
set -uo pipefail

exec >>/tmp/conlang_monitor_tiny32_n2560000_e1.log 2>&1

readonly gcloud_executable=/snap/bin/gcloud
readonly tpu_name=kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a
readonly project_name=he-vision-group
readonly tpu_zone=us-west4-a
readonly run_name=20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1
readonly launcher_log=/home/kristinelu/elf-runs/logs/${run_name}.launcher.log
readonly checkpoint_path=/home/kristinelu/elf-runs/${run_name}/checkpoint.msgpack

printf 'MONITOR_TIME=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${gcloud_executable}" compute tpus tpu-vm describe "${tpu_name}" \
    --project="${project_name}" \
    --zone="${tpu_zone}" \
    --format='value(state,health)'

"${gcloud_executable}" compute tpus tpu-vm ssh "${tpu_name}" \
    --project="${project_name}" \
    --zone="${tpu_zone}" \
    --worker=all \
    --command="set -uo pipefail; echo worker=\$(hostname); active_trainers=0; for process_identifier in \$(pgrep -x python || true); do process_arguments=\$(tr '\\000' ' ' < /proc/\${process_identifier}/cmdline || true); case \" \${process_arguments} \" in *\" train.py \"*|*\"/train.py \"*) echo trainer_pid=\${process_identifier}; active_trainers=1;; esac; done; echo active_trainers=\${active_trainers}; if test -f '${launcher_log}'; then grep -E '^(step=|TRAINING_)' '${launcher_log}' | tail -n 3 || true; else echo launcher_log=absent; fi; if test -f '${checkpoint_path}'; then stat -c 'checkpoint_bytes=%s checkpoint_mtime=%y' '${checkpoint_path}'; else echo checkpoint=absent; fi; for resource in /dev/vfio/* /tmp/libtpu_lockfile; do test -e \"\${resource}\" || continue; holders=\$(sudo -n fuser \"\${resource}\" 2>/dev/null || true); if test -n \"\${holders}\"; then echo holder_resource=\${resource} holder_pids=\${holders}; fi; done" \
    --quiet
