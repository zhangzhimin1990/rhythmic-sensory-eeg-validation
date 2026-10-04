import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_dryad_confirmatory_protocol import (
    DEFAULT_GATE,
    DEFAULT_PROTOCOL,
    validate,
)


class DryadConfirmatoryProtocolTests(unittest.TestCase):
    def test_frozen_protocol_matches_technical_gate(self):
        result = validate(DEFAULT_PROTOCOL, DEFAULT_GATE)
        self.assertTrue(result["protocol_valid"])
        self.assertEqual(result["n_confirmatory_subjects"], 35)
        self.assertEqual(result["technical_failures_excluded"], [1, 15, 25, 41])
        self.assertFalse(result["outcomes_accessed_by_validator"])

    def test_protocol_rejects_subject_drift(self):
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        changed = copy.deepcopy(protocol)
        changed["confirmatory_subjects"] = changed["confirmatory_subjects"][:-1]
        changed["n_confirmatory_subjects"] -= 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not equal"):
                validate(path, DEFAULT_GATE)

    def test_protocol_rejects_primary_metric_drift(self):
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        protocol["primary_neural_estimand"]["metric"] = "best_observed_metric"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "primary neural metric drifted"):
                validate(path, DEFAULT_GATE)

    def test_protocol_rejects_equivalence_bound_drift(self):
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        protocol["primary_neural_estimand"][
            "standardized_smallest_effect_of_interest_dz"
        ] = 0.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "equivalence bound drifted"):
                validate(path, DEFAULT_GATE)


if __name__ == "__main__":
    unittest.main()
