import unittest

from scripts.assemble_dryad_confirmatory_claim_release import decide


def endpoint(primary=True, behaviour=True, specific=True, within=True, prediction=False):
    return (
        {"branch": "p", "theta_plus_superiority_claim_unlocked": primary},
        {"branch": "b", "acute_task_performance_advantage_claim_unlocked": behaviour},
        {"branch": "c", "theta_plus_specific_coupling_supported": specific, "theta_plus_within_condition_coupling_supported": within},
        {"branch": "r", "prediction_increment_claim_unlocked": prediction},
    )


class DryadConfirmatoryClaimReleaseTests(unittest.TestCase):
    def test_full_concurrent_chain_still_does_not_unlock_causality(self):
        result = decide(*endpoint(prediction=True))
        self.assertEqual(result["claim_ceiling"], "target_engagement_behaviour_and_condition_specific_concurrent_coupling")
        self.assertFalse(result["causal_mechanism_claim_unlocked"])
        self.assertFalse(result["clinical_treatment_selection_claim_unlocked"])

    def test_target_and_behaviour_without_specific_coupling(self):
        result = decide(*endpoint(specific=False, within=True))
        self.assertEqual(result["claim_ceiling"], "target_engagement_and_behaviour_without_specific_coupling")

    def test_target_only(self):
        result = decide(*endpoint(behaviour=False, specific=False, within=False))
        self.assertEqual(result["claim_ceiling"], "target_engagement_without_behaviour_chain")

    def test_behaviour_without_primary_target(self):
        result = decide(*endpoint(primary=False, specific=False, within=False))
        self.assertEqual(result["claim_ceiling"], "behaviour_without_primary_target_engagement")

    def test_prediction_does_not_rescue_missing_chain(self):
        result = decide(*endpoint(primary=False, behaviour=False, specific=False, within=False, prediction=True))
        self.assertEqual(result["claim_ceiling"], "measurement_and_translation_boundary")
        self.assertTrue(result["prediction_cannot_raise_causal_claim_ceiling"])


if __name__ == "__main__":
    unittest.main()
