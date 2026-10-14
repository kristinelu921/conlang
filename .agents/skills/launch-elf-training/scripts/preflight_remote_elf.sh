#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 PROJECT TPU_NAME ZONE DATA_PATH OUTPUT_PATH REQUIRED_FREE_GIB" >&2
  exit 2
fi

project=$1
tpu_name=$2
zone=$3
data_path=$4
output_path=$5
required_free_gib=$6

if [[ $data_path == /dev/shm/elf-* ]]; then
  data_storage_root=/dev/shm
elif [[ $data_path == /home/* ]]; then
  data_storage_root=/home
else
  echo "Data must be below /home or an isolated /dev/shm/elf-* path." >&2
  exit 3
fi
if [[ $output_path != /home/* ]]; then
  echo "Output paths must be node-local paths below /home." >&2
  exit 3
fi
if [[ $data_path == *kmh-nfs* || $output_path == *kmh-nfs* ]]; then
  echo "Cross-zone NFS paths are forbidden." >&2
  exit 3
fi
if ! [[ $required_free_gib =~ ^[0-9]+$ ]]; then
  echo "REQUIRED_FREE_GIB must be a non-negative integer." >&2
  exit 2
fi

state=$(gcloud compute tpus tpu-vm describe "$tpu_name" \
  --project="$project" --zone="$zone" --format='value(state)')
health=$(gcloud compute tpus tpu-vm describe "$tpu_name" \
  --project="$project" --zone="$zone" --format='value(health)')
if [[ $state != READY || $health != HEALTHY ]]; then
  echo "TPU must be READY and HEALTHY; found state=$state health=$health." >&2
  exit 4
fi

remote_command="set -eu; test -z \"\$(sudo lsof -t -w /dev/vfio/0 2>/dev/null || true)\"; test -z \"\$(pgrep -f '[p]ython.*train.py' || true)\"; data_available=\$(df -BG --output=avail '$data_storage_root' | tail -n 1 | tr -dc '0-9'); output_available=\$(df -BG --output=avail / | tail -n 1 | tr -dc '0-9'); test \"\$data_available\" -ge $required_free_gib; test \"\$output_available\" -ge $required_free_gib; test ! -e '$output_path'; printf 'host=%s data_free_gib=%s output_free_gib=%s tpu_device=idle\\n' \"\$(hostname)\" \"\$data_available\" \"\$output_available\""
gcloud compute tpus tpu-vm ssh "$tpu_name" --project="$project" \
  --zone="$zone" --worker=all --command="$remote_command"
