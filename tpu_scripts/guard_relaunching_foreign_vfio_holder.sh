#!/usr/bin/env bash
set -euo pipefail

project_identifier=${1:?'usage: guard_relaunching_foreign_vfio_holder.sh PROJECT TPU_NAME ZONE EXPECTED_OWNER COMMAND_FRAGMENT [SECONDS]'}
tpu_identifier=${2:?'usage: guard_relaunching_foreign_vfio_holder.sh PROJECT TPU_NAME ZONE EXPECTED_OWNER COMMAND_FRAGMENT [SECONDS]'}
tpu_zone=${3:?'usage: guard_relaunching_foreign_vfio_holder.sh PROJECT TPU_NAME ZONE EXPECTED_OWNER COMMAND_FRAGMENT [SECONDS]'}
expected_owner=${4:?'usage: guard_relaunching_foreign_vfio_holder.sh PROJECT TPU_NAME ZONE EXPECTED_OWNER COMMAND_FRAGMENT [SECONDS]'}
command_fragment=${5:?'usage: guard_relaunching_foreign_vfio_holder.sh PROJECT TPU_NAME ZONE EXPECTED_OWNER COMMAND_FRAGMENT [SECONDS]'}
guard_duration_seconds=${6:-900}

if [[ "${tpu_identifier,,}" != *kristine* ]]; then
    printf 'ERROR: TPU name must contain kristine: %s\n' "${tpu_identifier}" >&2
    exit 2
fi

remote_guard_command=$(cat <<REMOTE_COMMAND
set -euo pipefail
guard_log=/home/kristinelu/elf-runs/foreign-vfio-guard.log
guard_pid=/home/kristinelu/elf-runs/foreign-vfio-guard.pid
if test -f "\${guard_pid}" && kill -0 "\$(cat "\${guard_pid}")" 2>/dev/null; then
    printf 'GUARD_ALREADY_ACTIVE pid=%s host=%s\\n' "\$(cat "\${guard_pid}")" "\$(hostname)"
    exit 0
fi
nohup bash -s -- '${expected_owner}' '${command_fragment}' '${guard_duration_seconds}' > "\${guard_log}" 2>&1 < /dev/null <<'GUARD_SCRIPT' &
set -euo pipefail
expected_owner=\$1
command_fragment=\$2
guard_duration_seconds=\$3
current_user=\$(id -un)
# The parent shell's elapsed-time counter can be inherited; use wall clock so
# a fresh guard always receives its complete bounded handoff interval.
deadline_epoch=\$((\$(date +%s) + guard_duration_seconds))
while (( \$(date +%s) < deadline_epoch )); do
    holder_processes=()
    for accelerator_resource in /dev/vfio/* /tmp/libtpu_lockfile; do
        [[ -e "\${accelerator_resource}" ]] || continue
        fuser_output=\$(sudo -n fuser "\${accelerator_resource}" 2>/dev/null || true)
        # fuser emits whitespace-separated PIDs and may omit the final newline.
        for process_identifier in \${fuser_output}; do
            [[ "\${process_identifier}" =~ ^[0-9]+\$ ]] || continue
            holder_processes+=("\${process_identifier}")
        done
    done
    if (( \${#holder_processes[@]} == 0 )); then
        sleep 1
        continue
    fi
    while read -r process_identifier; do
        process_owner=\$(ps -p "\${process_identifier}" -o user= | xargs || true)
        process_command=\$(ps -p "\${process_identifier}" -o args= || true)
        if [[ "\${process_owner}" == "\${current_user}" ]] \
            && [[ "\${process_command}" == *train.py* ]]; then
            printf 'GUARD_YIELD_ELF pid=%s\\n' "\${process_identifier}"
            exit 0
        fi
        if [[ "\${process_owner}" != "\${expected_owner}" ]] \
            || [[ "\${process_command}" != *"\${command_fragment}"* ]]; then
            printf 'GUARD_REFUSED pid=%s owner=%s\\n' "\${process_identifier}" "\${process_owner}" >&2
            exit 1
        fi
        sudo -n kill -KILL "\${process_identifier}"
        printf 'GUARD_KILLED_FOREIGN pid=%s\\n' "\${process_identifier}"
    done < <(printf '%s\\n' "\${holder_processes[@]}" | sort -nu)
    sleep 1
done
printf 'GUARD_TIMEOUT\\n' >&2
exit 1
GUARD_SCRIPT
printf '%s\\n' "\$!" > "\${guard_pid}"
printf 'GUARD_STARTED pid=%s host=%s\\n' "\$!" "\$(hostname)"
REMOTE_COMMAND
)

gcloud compute tpus tpu-vm ssh "${tpu_identifier}" \
    --project="${project_identifier}" \
    --zone="${tpu_zone}" \
    --worker=all \
    --command="${remote_guard_command}"
