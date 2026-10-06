from pathlib import Path
import tempfile
import unittest

from tools.cleanup_dut_outputs import (
    PROTECTED_EXPERIMENTS,
    _assert_direct_child,
    build_manifest,
)


class CleanupDutOutputsTests(unittest.TestCase):
    def test_manifest_protects_only_the_two_formal_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            output_root = tmp_path / "outputs"
            report_root = tmp_path / "reports"
            output_root.mkdir()
            report_root.mkdir()
            for name in (*PROTECTED_EXPERIMENTS, "obsolete_experiment"):
                (output_root / name).mkdir()
            (report_root / "old.csv").write_text("old", encoding="utf-8")

            text, outputs, reports = build_manifest(output_root, report_root)

            self.assertEqual(
                [path.name for path in outputs], ["obsolete_experiment"])
            self.assertEqual([path.name for path in reports], ["old.csv"])
            for name in PROTECTED_EXPERIMENTS:
                self.assertIn(str(output_root.resolve() / name), text)

    def test_deletion_guard_rejects_nested_target(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root = tmp_path / "root"
            nested = root / "candidate" / "nested"
            nested.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "non-direct deletion target"):
                _assert_direct_child(nested, root)


if __name__ == '__main__':
    unittest.main()
