"""Focused tests for the fixed FDCR/RDCF/SPDR test and report workflow."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tools import summarize_fdcr_rdcf_spdr as summarize


def make_summary(status='PASS'):
    records = []
    for index, name in enumerate(summarize.EXPERIMENTS, 1):
        is_hr = name.startswith('rtdetr_hrnetv2')
        baseline = 0.70 if is_hr else 0.60
        bonus = (0.00 if name.endswith('anti_uav') else
                 0.01 if name.endswith('fdcr') else
                 0.02 if name.endswith('rdcf') else 0.03)
        records.append({
            'index': index,
            'experiment': name,
            'status': status,
            'map_50_95': baseline + bonus,
            'map50': baseline + bonus + 0.20,
            'map75': baseline + bonus - 0.05,
            'ap_small': baseline + bonus - 0.10,
            'ap_medium': baseline + bonus - 0.04,
            'ap_large': baseline + bonus + 0.02,
            'ar_1': baseline + bonus - 0.20,
            'ar_10': baseline + bonus - 0.08,
            'ar_100': baseline + bonus - 0.03,
            'ar_small': baseline + bonus - 0.12,
            'ar_medium': baseline + bonus - 0.05,
            'ar_large': baseline + bonus + 0.01,
            'checkpoint': f'/outputs/{name}/best.pth',
            'config': f'/configs/{name}.yml',
            'message': '',
        })
    return {
        'split': 'test',
        'gpu': '1',
        'num_workers': 2,
        'require_ema': True,
        'records': records,
    }


class TestUnifiedComparison(unittest.TestCase):
    def test_gains_use_matching_backbone_baseline(self):
        payload = summarize.build_comparison(
            make_summary(), Path('/tmp/summary.json'))
        rows = {row['experiment']: row for row in payload['records']}

        pres_fdcr = rows['rtdetr_r18vd_dut_anti_uav_fdcr']
        self.assertEqual(
            pres_fdcr['baseline_experiment'], summarize.PRES_BASELINE)
        self.assertAlmostEqual(
            pres_fdcr['gains_vs_baseline']['AP'], 0.01)
        self.assertAlmostEqual(
            pres_fdcr['gains_vs_baseline_percentage_points']['APS'], 1.0)

        hr_spdr = rows['rtdetr_hrnetv2_w18_dut_anti_uav_spdr']
        self.assertEqual(hr_spdr['baseline_experiment'], summarize.HR_BASELINE)
        self.assertAlmostEqual(hr_spdr['gains_vs_baseline']['AP75'], 0.03)
        self.assertAlmostEqual(
            hr_spdr['gains_vs_baseline_percentage_points']['ARS'], 3.0)

        for baseline_name in (summarize.PRES_BASELINE,
                              summarize.HR_BASELINE):
            self.assertEqual(
                rows[baseline_name]['gains_vs_baseline']['AP'], 0.0)

    def test_generate_writes_json_and_complete_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / 'summary.json'
            summary_path.write_text(
                json.dumps(make_summary()), encoding='utf-8')
            report_dir = root / 'reports'

            payload, json_path, markdown_path = summarize.generate(
                summary_path, report_dir)

            self.assertTrue(payload['all_pass'])
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            saved = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['protocol']['input_size'], [640, 640])
            self.assertEqual(saved['protocol']['precision'], 'FP32')
            markdown = markdown_path.read_text(encoding='utf-8')
            for heading in ('AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1',
                            'AR10', 'AR100', 'ARS', 'ARM', 'ARL'):
                self.assertIn(heading, markdown)
            self.assertIn('Gains relative to the matching backbone baseline',
                          markdown)
            self.assertIn('| HRNetV2-W18 | SPDR |', markdown)

    def test_incomplete_evaluation_keeps_report_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_summary()
            data['records'][2]['status'] = 'FAILED'
            data['records'][2]['message'] = 'evaluation exited with code 1'
            summary_path = root / 'summary.json'
            summary_path.write_text(json.dumps(data), encoding='utf-8')

            with redirect_stdout(io.StringIO()):
                code = summarize.main(SimpleNamespace(
                    summary=str(summary_path), output_dir=str(root / 'out')))
            self.assertEqual(code, 1)
            markdown = (root / 'out' / summarize.DEFAULT_MARKDOWN_NAME)
            self.assertIn('Incomplete evaluations',
                          markdown.read_text(encoding='utf-8'))

    def test_missing_coco_metric_makes_report_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_summary()
            del data['records'][1]['ar_small']
            summary_path = root / 'summary.json'
            summary_path.write_text(json.dumps(data), encoding='utf-8')

            with redirect_stdout(io.StringIO()):
                code = summarize.main(SimpleNamespace(
                    summary=str(summary_path), output_dir=str(root / 'out')))
            self.assertEqual(code, 1)
            markdown = (root / 'out' / summarize.DEFAULT_MARKDOWN_NAME)
            self.assertIn('missing metrics: ARS',
                          markdown.read_text(encoding='utf-8'))

    def test_wrong_protocol_or_order_is_rejected(self):
        wrong_protocol = make_summary()
        wrong_protocol['split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'fixed test protocol'):
            summarize.build_comparison(
                wrong_protocol, Path('/tmp/summary.json'))

        wrong_order = make_summary()
        wrong_order['records'][0], wrong_order['records'][1] = (
            wrong_order['records'][1], wrong_order['records'][0])
        with self.assertRaisesRegex(ValueError, 'order/set mismatch'):
            summarize.build_comparison(
                wrong_order, Path('/tmp/summary.json'))


class TestShellWrapper(unittest.TestCase):
    def test_fixed_order_and_protocol_flags(self):
        script_path = (Path(__file__).resolve().parents[1] / 'tools' /
                       'test_fdcr_rdcf_spdr_best.sh')
        script = script_path.read_text(encoding='utf-8')
        positions = [script.index(f'"{name}"')
                     for name in summarize.EXPERIMENTS]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('GPU_ID=1', script)
        self.assertIn('NUM_WORKERS=2', script)
        self.assertIn('--split test', script)
        self.assertIn('--require-ema', script)
        self.assertIn('tools/test_all_best.py', script)
        self.assertIn('tools/summarize_fdcr_rdcf_spdr.py', script)
        self.assertNotIn('--amp', script)
        self.assertIn('test_status=$?', script)
        self.assertIn('summary_status', script)


if __name__ == '__main__':
    unittest.main()
