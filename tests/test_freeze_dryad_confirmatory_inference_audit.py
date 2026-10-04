import tempfile
import unittest
from pathlib import Path

from scripts.freeze_dryad_confirmatory_inference_audit import ASSETS, build_manifest, run


class DryadConfirmatoryInferenceAuditFreezeTests(unittest.TestCase):
    def test_manifest_covers_result_blind_extension_assets(self):
        manifest = build_manifest()
        self.assertEqual(manifest["n_assets"], len(ASSETS))
        self.assertIn("scripts/audit_dryad_confirmatory_group_inference.py", ASSETS)
        self.assertFalse(manifest["participant_level_derivation_changed"])
        self.assertFalse(manifest["primary_estimator_changed"])

    def test_freeze_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(output)


if __name__ == "__main__":
    unittest.main()
