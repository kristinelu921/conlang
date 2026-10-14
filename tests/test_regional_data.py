"""Tests for TPU-zone dataset bucket selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from regional_data import bucket_for_zone, dataset_uri_for_zone, region_from_zone, resolve_training_dataset


def test_zone_selects_only_the_bucket_in_that_region() -> None:
    """Each TPU zone should resolve to its own regional bucket."""
    assert region_from_zone("projects/123/zones/us-west4-a") == "us-west4"
    assert region_from_zone("asia-northeast1-b") == "asia-northeast1"
    assert bucket_for_zone("us-west4-a") == "kmh-gcp-us-west4"
    assert dataset_uri_for_zone("us-central1-a") == "gs://kmh-gcp-us-central1/kristine/datasets/conlangs"
    assert dataset_uri_for_zone("us-east5-c") == "gs://kmh-gcp-us-east5/kristine/datasets/conlangs"
    assert dataset_uri_for_zone("asia-northeast1-b") == "gs://kmh-gcp-asia-northeast1-b/kristine/datasets/conlangs"
    with pytest.raises(ValueError, match="no regional bucket"):
        bucket_for_zone("europe-west4-a")


def test_local_dataset_dir_skips_bucket_selection() -> None:
    """An explicit local directory should be used without reading a TPU zone."""
    local_dir = Path("datasets/linked_facts_v1")

    assert resolve_training_dataset(local_dir, zone="us-central1-a") == local_dir


def test_staging_uses_the_detected_region_and_checks_bucket_location(tmp_path: Path) -> None:
    """Staging should copy only after the bucket location matches the TPU region."""
    copied: list[str] = []

    def copy_dataset(uri: str, destination: Path) -> None:
        copied.append(uri)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").write_text("{}", encoding="utf-8")

    staged = resolve_training_dataset(
        None,
        zone="us-central1-b",
        cache_dir=tmp_path,
        bucket_location=lambda bucket: "US-CENTRAL1" if bucket == "kmh-gcp-us-central1" else "US-WEST4",
        copy_dataset=copy_dataset,
    )

    assert staged == tmp_path
    assert copied == ["gs://kmh-gcp-us-central1/kristine/datasets/conlangs"]


def test_mismatched_bucket_location_is_rejected(tmp_path: Path) -> None:
    """A bucket located outside the TPU region should not be copied."""

    def copy_dataset(uri: str, destination: Path) -> None:
        raise AssertionError(uri)

    with pytest.raises(ValueError, match="not the TPU region"):
        resolve_training_dataset(
            None,
            zone="us-east5-a",
            cache_dir=tmp_path,
            bucket_location=lambda bucket: "US-WEST4",
            copy_dataset=copy_dataset,
        )
