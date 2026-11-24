"""Verify benchmark evidence gates without model execution."""
from copy import deepcopy
import hashlib
import unittest
from benchmark_policy import verify_benchmark_samples


class BenchmarkPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = []
        self.hashes = {}
        self.metrics = {}
        for filter_name in ('strict-match', 'flexible-extract'):
            self.metrics['exact_match,' + filter_name] = .5
            for index in range(2):
                question = str(index)
                prompt_hash = hashlib.sha256(('Question: ' + question + '\nAnswer:').encode()).hexdigest()
                self.records.append({'doc_id': index, 'filter': filter_name, 'doc': {'question': question},
                    'doc_hash': 'document' + question, 'target_hash': 'target' + question,
                    'prompt_hash': prompt_hash, 'exact_match': index})
                self.hashes[str(index) + ':' + filter_name] = ['document' + question, prompt_hash, 'target' + question]

    def test_complete_identical_protocol_passes(self) -> None:
        for shots in (0, 5):
            result = verify_benchmark_samples(self.records, self.metrics, self.hashes, shots, 2)
            self.assertEqual(result['measurements'][0]['correct'], 1)

    def test_missing_or_duplicate_questions_fail(self) -> None:
        with self.assertRaises(ValueError):
            verify_benchmark_samples(self.records[:-1], self.metrics, self.hashes, 0, 2)
        records = deepcopy(self.records)
        records[1] = records[0]
        with self.assertRaises(ValueError):
            verify_benchmark_samples(records, self.metrics, self.hashes, 0, 2)

    def test_changed_prompt_target_or_document_fails(self) -> None:
        for field in ('prompt_hash', 'target_hash', 'doc_hash'):
            records = deepcopy(self.records)
            records[0][field] = 'changed'
            with self.assertRaises(ValueError):
                verify_benchmark_samples(records, self.metrics, self.hashes, 5, 2)

    def test_wrong_aggregate_fails(self) -> None:
        metrics = {**self.metrics, 'exact_match,strict-match': 1.0}
        with self.assertRaises(ValueError):
            verify_benchmark_samples(self.records, metrics, self.hashes, 0, 2)


if __name__ == '__main__':
    unittest.main()
