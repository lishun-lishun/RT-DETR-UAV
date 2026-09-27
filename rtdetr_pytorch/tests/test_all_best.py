"""Tests for sequential best-checkpoint evaluation orchestration."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools import test_all_best


COCO_OUTPUT = """
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.646
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.953
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.733
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.222
"""


def arguments(root, configs, report=None, dry_run=False):
    return SimpleNamespace(
        root=str(root), config_dir=str(configs), split='test', gpu='1',
        num_workers=2, report_dir=str(report) if report else None,
        dry_run=dry_run)


class TestAllBest(unittest.TestCase):
    def test_coco_metric_parser(self):
        metrics = test_all_best.parse_coco_metrics(COCO_OUTPUT)
        self.assertEqual(metrics['map_50_95'], 0.646)
        self.assertEqual(metrics['map50'], 0.953)
        self.assertEqual(metrics['map75'], 0.733)
        self.assertEqual(metrics['ap_small'], 0.545)
        self.assertEqual(metrics['ar_1'], 0.222)

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
