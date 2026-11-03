"""Choose the dataset bucket from the TPU zone and refuse every other region."""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

# The asia-northeast1 bucket name includes its original zone. Its location is still ASIA-NORTHEAST1.
REGIONAL_BUCKETS = {
    "asia-northeast1": "kmh-gcp-asia-northeast1-b",
    "us-central1": "kmh-gcp-us-central1",
    "us-central2": "kmh-gcp-us-central2",
    "us-east1": "kmh-gcp-us-east1",
    "us-east5": "kmh-gcp-us-east5",
    "us-west4": "kmh-gcp-us-west4",
}
DATASET_PREFIX = "kristine/datasets/conlangs"


def region_from_zone(zone: str) -> str:
    """Return the region that contains one TPU zone."""
    zone_name = zone.rstrip("/").split("/")[-1]
    region, separator, _area = zone_name.rpartition("-")
    if not separator or not region:
        raise ValueError(f"TPU zone {zone} does not name a region")
    return region


def bucket_for_zone(zone: str) -> str:
    """Return the one bucket whose location is this TPU zone's region."""
    region = region_from_zone(zone)
    try:
        return REGIONAL_BUCKETS[region]
    except KeyError:
        raise ValueError(f"no regional bucket for TPU zone {zone}") from None


def dataset_uri_for_zone(zone: str, dataset_prefix: str = DATASET_PREFIX) -> str:
    """Build the dataset URI in the bucket for this TPU zone."""
    if dataset_prefix.startswith("gs://") or dataset_prefix.startswith("/"):
        raise ValueError("dataset prefix must be an object path inside the regional bucket")
    return f"gs://{bucket_for_zone(zone)}/{dataset_prefix.strip('/')}"


def detect_tpu_zone() -> str:
    """Read the zone from the metadata server of a machine that has a TPU."""
    accelerator = _read_metadata("instance/attributes/accelerator-type")
    if not accelerator:
        raise ValueError("no TPU accelerator is attached, so no regional dataset bucket can be selected")
    zone = _read_metadata("instance/zone")
    if not zone:
        raise ValueError("the TPU metadata server did not report a zone")
    return zone


def resolve_training_dataset(
    dataset_dir: Path | None,
    dataset_prefix: str = DATASET_PREFIX,
    zone: str | None = None,
    cache_dir: Path | None = None,
    bucket_location: Callable[[str], str] | None = None,
    copy_dataset: Callable[[str, Path], None] | None = None,
    detect_zone: Callable[[], str] | None = None,
) -> Path:
    """Return a local dataset, or stage the copy stored in this TPU region's bucket."""
    if dataset_dir is not None:
        return dataset_dir
    selected_zone = (detect_zone or detect_tpu_zone)() if zone is None else zone
    bucket = bucket_for_zone(selected_zone)
    location = (bucket_location or gcloud_bucket_location)(bucket)
    region = region_from_zone(selected_zone)
    if location.lower().replace("_", "-") != region.lower():
        raise ValueError(f"bucket {bucket} is in {location}, which is not the TPU region {region}")
    destination = cache_dir if cache_dir is not None else Path.home() / ".cache" / "conlangs" / region / "datasets"
    if (destination / "manifest.json").is_file() and (destination / "train.jsonl.gz").is_file():
        return destination
    uri = dataset_uri_for_zone(selected_zone, dataset_prefix)
    (copy_dataset or gcloud_copy_dataset)(uri, destination)
    if not (destination / "manifest.json").is_file():
        raise FileNotFoundError(f"dataset {uri} is absent from the TPU region {region}")
    return destination


def gcloud_bucket_location(bucket: str) -> str:
    """Return the location reported for one bucket."""
    try:
        completed = subprocess.run(
            ["gcloud", "storage", "buckets", "describe", f"gs://{bucket}", "--format=value(location)"],
            check=True,
            capture_output=True,
            text=True,
        )
        location = completed.stdout.strip()
    except FileNotFoundError:
        location = _gsutil_bucket_location(bucket)
    except subprocess.CalledProcessError as error:
        if not _gcloud_storage_is_unavailable(error):
            raise
        location = _gsutil_bucket_location(bucket)
    if not location:
        raise ValueError(f"bucket {bucket} did not report a location")
    return location


def gcloud_copy_dataset(uri: str, destination: Path) -> None:
    """Copy one regional dataset prefix into a local directory."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", f"{uri}/*", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        subprocess.run(["gsutil", "-m", "cp", f"{uri}/*", str(destination)], check=True)
    except subprocess.CalledProcessError as error:
        if not _gcloud_storage_is_unavailable(error):
            raise
        subprocess.run(["gsutil", "-m", "cp", f"{uri}/*", str(destination)], check=True)


def _gcloud_storage_is_unavailable(error: subprocess.CalledProcessError) -> bool:
    """Recognize Cloud SDK releases that predate the gcloud storage surface."""
    assert isinstance(error.stderr, str)
    return "Invalid choice: 'storage'" in error.stderr


def _gsutil_bucket_location(bucket: str) -> str:
    """Read a bucket location with the legacy gsutil bundled on older TPU images."""
    completed = subprocess.run(
        ["gsutil", "ls", "-Lb", f"gs://{bucket}"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        label, separator, value = line.strip().partition(":")
        if separator and label == "Location constraint":
            return value.strip()
    raise ValueError(f"bucket {bucket} did not report a location")


def _read_metadata(path: str) -> str | None:
    """Read one GCE metadata value, or None when that key is absent."""
    request = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.read().decode().strip()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise ValueError("could not read the TPU zone from the instance metadata server") from error
    except urllib.error.URLError as error:
        raise ValueError("could not read the TPU zone from the instance metadata server") from error
