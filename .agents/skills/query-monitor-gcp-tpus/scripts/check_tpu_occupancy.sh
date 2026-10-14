#!/usr/bin/env bash
set -euo pipefail

project_identifier=${1:?'usage: check_tpu_occupancy.sh PROJECT TPU_ID ZONE [--claim]'}
tpu_identifier=${2:?'usage: check_tpu_occupancy.sh PROJECT TPU_ID ZONE [--claim]'}
tpu_zone=${3:?'usage: check_tpu_occupancy.sh PROJECT TPU_ID ZONE [--claim]'}
claim_mode=${4:-}
gcloud_ssh_flags=(--ssh-flag=-F --ssh-flag=/dev/null)
use_internal_tpu_ip=${ELF_TPU_USE_INTERNAL_IP:-1}

if [[ "${tpu_identifier,,}" != *kristine* ]]; then
    printf 'ERROR: refusing non-kristine TPU ID: %s\n' "${tpu_identifier}" >&2
    exit 1
fi
if [[ "$use_internal_tpu_ip" != 0 && "$use_internal_tpu_ip" != 1 ]]; then
    printf 'ERROR: ELF_TPU_USE_INTERNAL_IP must be 0 or 1\n' >&2
    exit 1
fi
if [[ "$use_internal_tpu_ip" == 1 ]]; then
    gcloud_ssh_flags+=(--internal-ip)
fi
if [[ ! "$tpu_identifier" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'ERROR: TPU ID contains unsupported characters: %s\n' "$tpu_identifier" >&2
    exit 1
fi
if [[ -n "${claim_mode}" && "${claim_mode}" != '--claim' ]]; then
    printf 'ERROR: optional fourth argument must be --claim\n' >&2
    exit 1
fi

current_state=$(
    gcloud compute tpus tpu-vm describe "${tpu_identifier}" \
        --project="${project_identifier}" \
        --zone="${tpu_zone}" \
        --format='value(state)'
)
if [[ "${current_state}" != 'READY' ]]; then
    printf 'UNKNOWN TPU_ID=%s ZONE=%s STATE=%s\n' \
        "${tpu_identifier}" "${tpu_zone}" "${current_state}"
    exit 4
fi

read -r -d '' inspect_command <<'REMOTE_COMMAND' || true
set -euo pipefail
current_user=$(id -un)
if ! command -v fuser >/dev/null || ! sudo -n true 2>/dev/null; then
    printf 'UNKNOWN worker=%s reason=vfio_inspection_unavailable\n' "$(hostname)"
    exit 4
fi
declare -a holder_processes=()
lockfile_holder_found=0
accelerator_resources=(/dev/vfio/*)
if [[ -e /tmp/libtpu_lockfile ]]; then
    accelerator_resources+=(/tmp/libtpu_lockfile)
fi
for accelerator_resource in "${accelerator_resources[@]}"; do
    [[ -e "${accelerator_resource}" ]] || continue
    if ! device_holders=$(sudo -n fuser "${accelerator_resource}" 2>/dev/null); then
        device_holders=''
    fi
    for process_identifier in ${device_holders}; do
        [[ "${process_identifier}" =~ ^[0-9]+$ ]] || continue
        holder_processes+=("${process_identifier}")
        if [[ "${accelerator_resource}" == /tmp/libtpu_lockfile ]]; then
            lockfile_holder_found=1
        fi
    done
done

protected_holder_found=0
training_entrypoint=train.py
while read -r process_identifier process_owner process_name process_arguments; do
    [[ "${process_identifier}" =~ ^[0-9]+$ ]] || continue
    padded_process_arguments=" ${process_arguments} "
    if [[ "${process_name,,}" == python* \
            && ("${padded_process_arguments}" == *" ${training_entrypoint} "* \
                || "${padded_process_arguments}" == *"/${training_entrypoint} "*) ]]; then
        protected_holder_found=1
        printf 'PROTECTED worker=%s pid=%s user=%s reason=active_train_process command=%q\n' \
            "$(hostname)" "${process_identifier}" "${process_owner:-unknown}" \
            "${process_arguments}"
    fi
done < <(ps -eo pid=,user=,comm=,args=)

if [[ -e /tmp/libtpu_lockfile ]] && (( lockfile_holder_found == 0 )); then
    lockfile_metadata=$(sudo -n stat -c 'owner=%U uid=%u mode=%a size=%s modified=%y' \
        /tmp/libtpu_lockfile 2>/dev/null || true)
    protected_holder_found=1
    printf 'PROTECTED worker=%s reason=stale_libtpu_lock %s\n' \
        "$(hostname)" "${lockfile_metadata}"
fi

if (( ${#holder_processes[@]} == 0 && protected_holder_found == 0 )); then
    printf 'AVAILABLE worker=%s\n' "$(hostname)"
    exit 0
fi

if (( ${#holder_processes[@]} > 0 )); then
    mapfile -t unique_holder_processes < <(printf '%s\n' "${holder_processes[@]}" | sort -nu)
    for process_identifier in "${unique_holder_processes[@]}"; do
        process_owner=$(ps -o user= -p "${process_identifier}" | xargs)
        process_arguments=$(ps -o args= -p "${process_identifier}" || true)
        process_working_directory=$(sudo -n readlink "/proc/${process_identifier}/cwd" 2>/dev/null || true)
        sanitized_process_arguments=$(sed -E \
            's/kmh-tpuvm-[A-Za-z0-9._-]*kristine[A-Za-z0-9._-]*//g' \
            <<<"$process_arguments")
        sanitized_working_directory=$(sed -E \
            's/kmh-tpuvm-[A-Za-z0-9._-]*kristine[A-Za-z0-9._-]*//g' \
            <<<"$process_working_directory")
        searchable_process="${process_owner} ${sanitized_process_arguments} ${sanitized_working_directory}"
        if [[ "${process_owner}" == "${current_user}" || "${searchable_process,,}" == *kristine* ]]; then
            protected_holder_found=1
            printf 'PROTECTED worker=%s pid=%s user=%s command=%q\n' \
                "$(hostname)" "${process_identifier}" "${process_owner}" "${process_arguments}"
        else
            printf 'FOREIGN worker=%s pid=%s user=%s command=%q\n' \
                "$(hostname)" "${process_identifier}" "${process_owner}" "${process_arguments}"
        fi
    done
fi

if (( protected_holder_found )); then
    printf 'OCCUPIED worker=%s\n' "$(hostname)"
else
    printf 'RECLAIMABLE worker=%s\n' "$(hostname)"
fi
REMOTE_COMMAND

if ! inspection_output=$(gcloud compute tpus tpu-vm ssh "${tpu_identifier}" \
        "${gcloud_ssh_flags[@]}" \
        --project="${project_identifier}" \
        --zone="${tpu_zone}" \
        --worker=all \
        --command="${inspect_command}"); then
    printf '%s\n' "${inspection_output}"
    printf 'UNKNOWN TPU_ID=%s ZONE=%s reason=ssh_or_inspection_failed\n' \
        "${tpu_identifier}" "${tpu_zone}"
    exit 4
fi
printf '%s\n' "${inspection_output}"

if grep -q 'OCCUPIED worker=' <<<"${inspection_output}"; then
    exit 2
fi
if grep -q 'UNKNOWN worker=' <<<"${inspection_output}"; then
    exit 4
fi
if ! grep -q 'RECLAIMABLE worker=' <<<"${inspection_output}"; then
    exit 0
fi
if [[ "${claim_mode}" != '--claim' ]]; then
    exit 3
fi

read -r -d '' claim_command <<'REMOTE_COMMAND' || true
set -euo pipefail
current_user=$(id -un)
if ! command -v fuser >/dev/null || ! sudo -n true 2>/dev/null; then
    printf 'UNKNOWN worker=%s reason=vfio_inspection_unavailable\n' "$(hostname)"
    exit 4
fi
declare -a holder_processes=()
lockfile_holder_found=0
accelerator_resources=(/dev/vfio/*)
if [[ -e /tmp/libtpu_lockfile ]]; then
    accelerator_resources+=(/tmp/libtpu_lockfile)
fi
for accelerator_resource in "${accelerator_resources[@]}"; do
    [[ -e "${accelerator_resource}" ]] || continue
    if ! device_holders=$(sudo -n fuser "${accelerator_resource}" 2>/dev/null); then
        device_holders=''
    fi
    for process_identifier in ${device_holders}; do
        [[ "${process_identifier}" =~ ^[0-9]+$ ]] || continue
        holder_processes+=("${process_identifier}")
        if [[ "${accelerator_resource}" == /tmp/libtpu_lockfile ]]; then
            lockfile_holder_found=1
        fi
    done
done

training_entrypoint=train.py
while read -r process_identifier process_owner process_name process_arguments; do
    [[ "${process_identifier}" =~ ^[0-9]+$ ]] || continue
    padded_process_arguments=" ${process_arguments} "
    if [[ "${process_name,,}" == python* \
            && ("${padded_process_arguments}" == *" ${training_entrypoint} "* \
                || "${padded_process_arguments}" == *"/${training_entrypoint} "*) ]]; then
        printf 'OCCUPIED worker=%s pid=%s user=%s reason=active_train_process\n' \
            "$(hostname)" "${process_identifier}" "${process_owner:-unknown}"
        exit 2
    fi
done < <(ps -eo pid=,user=,comm=,args=)

if [[ -e /tmp/libtpu_lockfile ]] && (( lockfile_holder_found == 0 )); then
    lockfile_metadata=$(sudo -n stat -c 'owner=%U uid=%u mode=%a size=%s modified=%y' \
        /tmp/libtpu_lockfile 2>/dev/null || true)
    printf 'OCCUPIED worker=%s reason=stale_libtpu_lock %s\n' \
        "$(hostname)" "${lockfile_metadata}"
    exit 2
fi

if (( ${#holder_processes[@]} == 0 )); then
    printf 'AVAILABLE worker=%s\n' "$(hostname)"
    exit 0
fi

mapfile -t unique_holder_processes < <(printf '%s\n' "${holder_processes[@]}" | sort -nu)
declare -A holder_start_times=()
for process_identifier in "${unique_holder_processes[@]}"; do
    process_owner=$(ps -o user= -p "${process_identifier}" | xargs)
    process_arguments=$(ps -o args= -p "${process_identifier}" || true)
    process_working_directory=$(sudo -n readlink "/proc/${process_identifier}/cwd" 2>/dev/null || true)
    holder_start_times["${process_identifier}"]=$(awk '{print $22}' "/proc/${process_identifier}/stat")
    sanitized_process_arguments=$(sed -E \
        's/kmh-tpuvm-[A-Za-z0-9._-]*kristine[A-Za-z0-9._-]*//g' \
        <<<"$process_arguments")
    sanitized_working_directory=$(sed -E \
        's/kmh-tpuvm-[A-Za-z0-9._-]*kristine[A-Za-z0-9._-]*//g' \
        <<<"$process_working_directory")
    searchable_process="${process_owner} ${sanitized_process_arguments} ${sanitized_working_directory}"
    if [[ "${process_owner}" == "${current_user}" || "${searchable_process,,}" == *kristine* ]]; then
        printf 'OCCUPIED worker=%s pid=%s user=%s\n' \
            "$(hostname)" "${process_identifier}" "${process_owner}"
        exit 2
    fi
done

for process_identifier in "${unique_holder_processes[@]}"; do
    current_start_time=$(awk '{print $22}' "/proc/${process_identifier}/stat" 2>/dev/null || true)
    if [[ -n "${current_start_time}" && "${current_start_time}" == "${holder_start_times[$process_identifier]}" ]]; then
        sudo -n kill -TERM -- "${process_identifier}"
    fi
done
for _ in 1 2 3 4 5; do
    remaining_holder_found=0
    for process_identifier in "${unique_holder_processes[@]}"; do
        if kill -0 "${process_identifier}" 2>/dev/null; then
            remaining_holder_found=1
        fi
    done
    (( remaining_holder_found == 0 )) && break
    sleep 1
done
for process_identifier in "${unique_holder_processes[@]}"; do
    current_start_time=$(awk '{print $22}' "/proc/${process_identifier}/stat" 2>/dev/null || true)
    process_owner=$(ps -o user= -p "${process_identifier}" | xargs || true)
    process_arguments=$(ps -o args= -p "${process_identifier}" || true)
    process_working_directory=$(sudo -n readlink "/proc/${process_identifier}/cwd" 2>/dev/null || true)
    sanitized_process_arguments=$(sed -E \
        's/kmh-tpuvm-[A-Za-z0-9._-]*kristine[A-Za-z0-9._-]*//g' \
        <<<"$process_arguments")
    sanitized_working_directory=$(sed -E \
        's/kmh-tpuvm-[A-Za-z0-9._-]*kristine[A-Za-z0-9._-]*//g' \
        <<<"$process_working_directory")
    searchable_process="${process_owner} ${sanitized_process_arguments} ${sanitized_working_directory}"
    if [[ -n "${current_start_time}" \
        && "${current_start_time}" == "${holder_start_times[$process_identifier]}" \
        && "${process_owner}" != "${current_user}" \
        && "${searchable_process,,}" != *kristine* ]]; then
        sudo -n kill -KILL -- "${process_identifier}"
    fi
done
printf 'CLAIMED worker=%s\n' "$(hostname)"
REMOTE_COMMAND

if ! claim_output=$(gcloud compute tpus tpu-vm ssh "${tpu_identifier}" \
        "${gcloud_ssh_flags[@]}" \
        --project="${project_identifier}" \
        --zone="${tpu_zone}" \
        --worker=all \
        --command="${claim_command}"); then
    printf '%s\n' "${claim_output}"
    if grep -q 'OCCUPIED worker=' <<<"${claim_output}"; then
        exit 2
    fi
    printf 'UNKNOWN TPU_ID=%s ZONE=%s reason=claim_failed\n' \
        "${tpu_identifier}" "${tpu_zone}"
    exit 4
fi
printf '%s\n' "${claim_output}"

if grep -q 'OCCUPIED worker=' <<<"${claim_output}"; then
    exit 2
fi
