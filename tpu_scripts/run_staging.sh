#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s RUN_ID\n' "$0" >&2
    exit 2
fi

repository_root=${ELF_STAGING_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
staging_root=${ELF_STAGING_ROOT:-/kmh-nfs-ssd-us-mount/staging/kristine/elf}
run_identifier=$1
selected_config=${ELF_STAGING_CONFIG:-}

if [[ ! -f "${repository_root}/train.py" ]]; then
    printf 'ERROR: conlang source is missing train.py: %s\n' "${repository_root}" >&2
    exit 1
fi
if [[ ! "$run_identifier" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'ERROR: RUN_ID contains unsupported characters\n' >&2
    exit 2
fi
if [[ -n "$selected_config" && ("$selected_config" == /* || "$selected_config" == *..*) ]]; then
    printf 'ERROR: ELF_STAGING_CONFIG must be a repository-relative path without ..\n' >&2
    exit 2
fi
if [[ -n "$selected_config" && ! -f "${repository_root}/${selected_config}" ]]; then
    printf 'ERROR: selected config is missing: %s\n' "$selected_config" >&2
    exit 1
fi

snapshot_timestamp=$(date -u +'%Y%m%dT%H%M%S.%NZ')
staging_directory="${staging_root}/${run_identifier}-${snapshot_timestamp}"

cleanup_partial_snapshot() {
    local exit_status=$?

    trap - EXIT
    if (( exit_status != 0 )); then
        rm -rf -- "$staging_directory"
        printf 'ELF_ROLLBACK_LOCAL_STAGING=REMOVED path=%s\n' "$staging_directory" >&2
    fi
    exit "$exit_status"
}

umask 002
mkdir -p "$staging_root"
mkdir "$staging_directory"
trap cleanup_partial_snapshot EXIT

rsync -rl \
    --no-owner \
    --no-group \
    --no-perms \
    --omit-dir-times \
    --exclude='.git' \
    --exclude='.agents/' \
    --exclude='.codex/' \
    --exclude='.netrc' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='datasets/' \
    --exclude='experiment_log/' \
    --exclude='wandb/' \
    --exclude='outputs/' \
    --exclude='*.pyc' \
    "${repository_root}/" \
    "${staging_directory}/"

if [[ ! -f "${staging_directory}/train.py" ]]; then
    printf 'ERROR: staging verification failed: train.py was not copied\n' >&2
    exit 1
fi
if [[ -n "$selected_config" && ! -f "${staging_directory}/${selected_config}" ]]; then
    printf 'ERROR: staging verification failed: selected config was not copied\n' >&2
    exit 1
fi

trap - EXIT
printf 'STAGEDIR=%s\n' "$staging_directory"
