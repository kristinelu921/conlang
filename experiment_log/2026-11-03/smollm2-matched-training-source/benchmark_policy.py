"""Verify full-test coverage, exact-match aggregation and immutable prompt fingerprints."""
from __future__ import annotations
import hashlib
import math
from typing import Any


def verify_benchmark_samples(records: list[dict[str, Any]], metrics: dict[str, Any],
                             baseline_hashes: dict[str, list[str]], shots: int,
                             expected_count: int = 1319) -> dict[str, Any]:
    """Fail closed if any test question, target, prompt, coverage or reported score changes."""
    if shots not in (0, 5) or len(records) != expected_count * 2:
        raise ValueError('Unexpected benchmark shot count or sample count.')
    measurements = []
    for filter_name in ('strict-match', 'flexible-extract'):
        selected = [record for record in records if record['filter'] == filter_name]
        if len(selected) != expected_count or {record['doc_id'] for record in selected} != set(range(expected_count)):
            raise ValueError('Benchmark question coverage is incomplete or duplicated.')
        for record in selected:
            expected = baseline_hashes[str(record['doc_id']) + ':' + filter_name]
            prompt_hash = expected[1] if shots == 5 else hashlib.sha256(
                ('Question: ' + record['doc']['question'] + '\nAnswer:').encode()).hexdigest()
            if [record['doc_hash'], record['prompt_hash'], record['target_hash']] != [expected[0], prompt_hash, expected[2]]:
                raise ValueError('Benchmark document, target or prompt fingerprint changed.')
        correct = sum(int(record['exact_match']) for record in selected)
        if not math.isclose(metrics['exact_match,' + filter_name], correct / expected_count, abs_tol=1e-12):
            raise ValueError('Aggregate accuracy differs from the per-question outcomes.')
        measurements.append({'filter': filter_name, 'correct': correct, 'examples': expected_count,
            'accuracy': correct / expected_count})
    return {'status': 'passed', 'coverage_verified': True, 'baseline_prompt_hashes_match': True,
            'measurements': measurements}
