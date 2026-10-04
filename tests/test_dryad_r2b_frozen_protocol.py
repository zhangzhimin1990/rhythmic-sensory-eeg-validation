import json
import unittest

from scripts.develop_dryad_r2b_spatial_qc import (
    DEVELOPMENT_BRIDGE_CORRELATION_CEILING,
    DEVELOPMENT_NEIGHBOUR_CORRELATION_FLOOR,
    DEVELOPMENT_ROBUST_Z_LIMIT,
)
from scripts.develop_dryad_r2b_technical_subject import (
    EXPECTED_TRIALS,
    MAXIMUM_BAD_CHANNELS,
    MINIMUM_EVENT_COVERAGE,
    MINIMUM_GOOD_CHANNELS,
    MINIMUM_OVERALL_USABLE,
)
from scripts.freeze_dryad_r2b_protocol import ROOT


class DryadR2bFrozenProtocolTests(unittest.TestCase):
    def test_frozen_config_matches_implementation_constants(self):
        config = json.loads(
            (ROOT / "configs/dryad_r2b_protocol_frozen_v1.json").read_text(encoding="utf-8")
        )
        channel = config["channel_rules"]
        event = config["event_and_epoch_rules"]
        self.assertEqual(channel["robust_z_limit"], DEVELOPMENT_ROBUST_Z_LIMIT)
        self.assertEqual(
            channel["neighbour_correlation_floor"], DEVELOPMENT_NEIGHBOUR_CORRELATION_FLOOR
        )
        self.assertEqual(
            channel["bridge_correlation_ceiling"], DEVELOPMENT_BRIDGE_CORRELATION_CEILING
        )
        self.assertEqual(channel["maximum_bad_channels"], MAXIMUM_BAD_CHANNELS)
        self.assertEqual(channel["minimum_good_channels"], MINIMUM_GOOD_CHANNELS)
        self.assertEqual(event["expected_trials"], EXPECTED_TRIALS)
        self.assertEqual(event["minimum_event_coverage_fraction"], MINIMUM_EVENT_COVERAGE)
        self.assertEqual(event["minimum_overall_epoch_usable_fraction"], MINIMUM_OVERALL_USABLE)


if __name__ == "__main__":
    unittest.main()
