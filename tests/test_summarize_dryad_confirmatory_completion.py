import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_dryad_confirmatory_completion import build_ledger


ROOT = Path(__file__).resolve().parents[1]


class DryadConfirmatoryCompletionLedgerTests(unittest.TestCase):
    def build_synthetic_subject_tree(self, directory: Path) -> tuple[Path, Path]:
        source_protocol = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
        protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
        protocol_path = directory / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        subject_root = directory / "subjects"
        for subject in map(int, protocol["confirmatory_subjects"]):
            subject_dir = subject_root / f"S{subject}"
            subject_dir.mkdir(parents=True)
            digest = f"{subject:064x}"
            (subject_dir / "confirmatory_run_summary.json").write_text(
                json.dumps({
                    "status": "confirmatory_subject_complete",
                    "source_bdf_sha256": digest,
                    "temporary_bdf_deleted_after_verification": True,
                    "individual_theta_hz": 5.0,
                    "post_freeze_amendment_ids": [],
                }),
                encoding="utf-8",
            )
            (subject_dir / "confirmatory_subject_summary.json").write_text(
                json.dumps({
                    "status": "confirmatory_subject_derivation_complete",
                    "source_bdf_sha256": digest,
                    "analysis_frequency_hz": 5.0,
                    "n_technical_usable_epochs_reproduced": 160,
                    "n_good_frozen_qc_channels": 64,
                    "frontocentral_roi": ["Fz", "FCz", "F1", "F2"],
                }),
                encoding="utf-8",
            )
        return protocol_path, subject_root

    def test_ledger_has_exact_locked_cohort_and_no_effect_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol, subjects = self.build_synthetic_subject_tree(Path(temporary))
            ledger = build_ledger(protocol, subjects)
        self.assertEqual(len(ledger), 35)
        self.assertEqual(ledger["subject"].nunique(), 35)
        forbidden = ("itpc", "snr", "accuracy", "reaction", "difference", "effect", "p_value")
        self.assertFalse(any(any(token in column.lower() for token in forbidden) for column in ledger.columns))

    def test_completed_rows_have_verified_cleanup_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol, subjects = self.build_synthetic_subject_tree(Path(temporary))
            ledger = build_ledger(protocol, subjects)
        complete = ledger.loc[ledger["status"].eq("complete")]
        self.assertGreaterEqual(len(complete), 15)
        self.assertTrue(complete["temporary_bdf_deleted"].all())
        self.assertTrue(complete["source_bdf_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())
        self.assertTrue(complete["frontocentral_roi_channels"].ge(4).all())


if __name__ == "__main__":
    unittest.main()
