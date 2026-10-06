"""Tests for sequential best-checkpoint evaluation orchestration."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools import test_all_best
from tools import test_dut


COCO_OUTPUT = """
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.646
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.953
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.733
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.222
"""


def arguments(root, configs, report=None, dry_run=False, experiments=None):
    return SimpleNamespace(
        root=str(root), config_dir=str(configs), split='test', gpu='1',
        num_workers=2, report_dir=str(report) if report else None,
        dry_run=dry_run, experiments=experiments, require_ema=False)


class TestAllBest(unittest.TestCase):
    def test_coco_metric_parser(self):
        metrics = test_all_best.parse_coco_metrics(COCO_OUTPUT)
        self.assertEqual(metrics['map_50_95'], 0.646)
        self.assertEqual(metrics['map50'], 0.953)
        self.assertEqual(metrics['map75'], 0.733)
        self.assertEqual(metrics['ap_small'], 0.545)
        self.assertEqual(metrics['ar_1'], 0.222)

    def test_ema_checkpoint_preflight_accepts_valid_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / 'valid.pth'
            missing = Path(directory) / 'missing.pth'
            torch.save({'ema': {'module': {'weight': torch.ones(1)}}}, valid)
            torch.save({'model': {'weight': torch.ones(1)}}, missing)
            with redirect_stdout(io.StringIO()):
                test_dut.validate_ema_checkpoint(valid)
            with self.assertRaisesRegex(ValueError, 'ema.module'):
                test_dut.validate_ema_checkpoint(missing)

    def test_dry_run_records_plan_without_creating_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'outputs'
            configs = Path(directory) / 'configs'
            root.mkdir()
            configs.mkdir()
            (root / 'model_a').mkdir()
            (root / 'model_a' / 'best.pth').touch()
            (configs / 'model_a.yml').touch()
            report = Path(directory) / 'report'
            with redirect_stdout(io.StringIO()):
                code = test_all_best.main(
                    arguments(root, configs, report, dry_run=True))
            self.assertEqual(code, 0)
            self.assertFalse(report.exists())

    def test_explicit_experiments_preserve_order_and_record_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'outputs'
            configs = Path(directory) / 'configs'
            root.mkdir()
            configs.mkdir()
            for name in ('model_a', 'model_b'):
                (configs / f'{name}.yml').touch()
            (root / 'model_a').mkdir()
            (root / 'model_a' / 'best.pth').touch()
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = test_all_best.main(arguments(
                    root, configs, dry_run=True,
                    experiments=['model_b', 'model_a']))
            output = stream.getvalue()
            self.assertEqual(code, 0)
            self.assertLess(output.index('model_b'), output.index('model_a'))
            self.assertIn('[MISSING_BEST] model_b', output)
            self.assertIn('[RUN] model_a', output)

    def test_fixed_list_dry_run_allows_output_root_not_created_yet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'future_outputs'
            configs = Path(directory) / 'configs'
            configs.mkdir()
            (configs / 'model_a.yml').touch()
            with redirect_stdout(io.StringIO()):
                code = test_all_best.main(arguments(
                    root, configs, dry_run=True,
                    experiments=['model_a']))
            self.assertEqual(code, 0)
            self.assertFalse(root.exists())

    def test_success_writes_csv_json_markdown_and_log_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'outputs'
            configs = Path(directory) / 'configs'
            root.mkdir()
            configs.mkdir()
            (root / 'model_a').mkdir()
            (root / 'model_a' / 'best.pth').touch()
            (configs / 'model_a.yml').touch()
            report = Path(directory) / 'report'

            def fake_run(command, environment, log_path):
                log_path.write_text(COCO_OUTPUT, encoding='utf-8')
                return 0, COCO_OUTPUT

            with patch.object(test_all_best, 'run_and_tee', fake_run), \
                    redirect_stdout(io.StringIO()):
                code = test_all_best.main(arguments(root, configs, report))
            self.assertEqual(code, 0)
            self.assertTrue((report / 'summary.csv').is_file())
            self.assertTrue((report / 'summary.json').is_file())
            summary = (report / 'summary.md').read_text(encoding='utf-8')
            self.assertIn('| model_a | PASS | 0.6460 | 0.9530 | 0.7330 |',
                          summary)


if __name__ == '__main__':
    unittest.main()
