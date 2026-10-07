from pathlib import Path
import tempfile
import unittest

from tools.cleanup_unused_outputs import (
    ACTIVE_EXPERIMENTS,
    build_plan,
    format_manifest,
    remove_direct_directory,
)


class CleanupUnusedOutputsTests(unittest.TestCase):
    def test_plan_keeps_active_names_and_only_deletes_other_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (*ACTIVE_EXPERIMENTS, "old_acr", "old_dgfr"):
                (root / name).mkdir()
            note = root / "README.txt"
            note.write_text("keep me", encoding="utf-8")

            keep_names = set(ACTIVE_EXPERIMENTS)
            plan = build_plan(root, keep_names)

            self.assertEqual(
                {path.name for path in plan["keep_existing"]}, keep_names)
            self.assertEqual(
                [path.name for path in plan["delete"]], ["old_acr", "old_dgfr"])
            self.assertEqual(plan["ignored_files"], [note])
            self.assertIn("delete_count: 2", format_manifest(root, keep_names, plan))

    def test_missing_active_names_are_reported_but_never_delete_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ACTIVE_EXPERIMENTS[0]).mkdir()
            plan = build_plan(root, set(ACTIVE_EXPERIMENTS))
            self.assertEqual(plan["delete"], [])
            self.assertEqual(len(plan["keep_missing"]), len(ACTIVE_EXPERIMENTS) - 1)

    def test_removal_is_limited_to_a_direct_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obsolete = root / "obsolete"
            nested = obsolete / "nested"
            nested.mkdir(parents=True)
            remove_direct_directory(obsolete, root)
            self.assertFalse(obsolete.exists())

            outside = root.parent / f"{root.name}_outside"
            outside.mkdir()
            try:
                with self.assertRaisesRegex(RuntimeError, "non-direct"):
                    remove_direct_directory(outside, root)
                self.assertTrue(outside.exists())
            finally:
                outside.rmdir()


if __name__ == "__main__":
    unittest.main()
