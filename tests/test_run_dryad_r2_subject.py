import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import shutil

import pandas as pd

from scripts.run_dryad_r2_subject import (
    load_frozen_subject,
    require_fresh_output,
    require_scratch_capacity,
    sha256,
    verify_retained_bdf_and_audit,
)


class DryadR2SubjectRunnerTests(unittest.TestCase):
    def test_load_frozen_subject_accepts_only_remaining_participants(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary) / "selection.csv"
            pd.DataFrame(
                {
                    "subject": [2, 13, 27, 23, 37],
                    "individual_freq": [5.01, 2.47, 3.15, 2.86, 3.35],
                    "uncompressed_bytes": [739, 771, 708, 785, 718],
                    "selection_order": [1, 2, 3, 4, 5],
                }
            ).to_csv(selection, index=False)
            frozen = load_frozen_subject(selection, 13)
            self.assertEqual(frozen["selection_order"], 2)
            self.assertEqual(frozen["individual_theta_hz"], 2.47)
            with self.assertRaisesRegex(ValueError, "remaining frozen R2"):
                load_frozen_subject(selection, 2)

    def test_nonempty_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "S13"
            audit = base / "audit"
            audit.mkdir(parents=True)
            (audit / "partial.csv").write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty output"):
                require_fresh_output(base)

    def test_scratch_capacity_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "scripts.run_dryad_r2_subject.shutil.disk_usage",
                return_value=shutil._ntuple_diskusage(1_000, 900, 100),
            ):
                with self.assertRaisesRegex(OSError, "insufficient scratch space"):
                    require_scratch_capacity(Path(temporary), 80, reserve_bytes=40)

    def test_scratch_capacity_records_preflight_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "scripts.run_dryad_r2_subject.shutil.disk_usage",
                return_value=shutil._ntuple_diskusage(1_000, 700, 300),
            ):
                result = require_scratch_capacity(
                    Path(temporary), 200, reserve_bytes=50
                )
        self.assertEqual(result["scratch_free_bytes_before_extraction"], 300)
        self.assertEqual(result["scratch_required_bytes"], 250)

    def test_resume_requires_bdf_hash_to_match_completed_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bdf = root / "S13_Stim.bdf"
            bdf.write_bytes(b"verified-retained-bdf")
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "trials.csv").write_text(
                "trial_index\n0\n", encoding="utf-8"
            )
            (audit_dir / "audit_summary.json").write_text(
                json.dumps(
                    {
                        "file_size_bytes": bdf.stat().st_size,
                        "source_sha256": sha256(bdf),
                        "n_scalp_eeg_channels": 64,
                    }
                ),
                encoding="utf-8",
            )
            verify_retained_bdf_and_audit(
                13, bdf.stat().st_size, bdf, audit_dir
            )
            bdf.write_bytes(b"tampered-retained-bdf")
            with self.assertRaises(RuntimeError):
                verify_retained_bdf_and_audit(
                    13, bdf.stat().st_size, bdf, audit_dir
                )


if __name__ == "__main__":
    unittest.main()
