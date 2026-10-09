"""Tests for the fixed PCX/ESDR/PSCA evaluation and report workflow."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tools import summarize_pcx_esdr_psca as summarize


METRIC_FIELDS = (
    'map_50_95', 'map50', 'map75',
    'ap_small', 'ap_medium', 'ap_large',
    'ar_1', 'ar_10', 'ar_100',
    'ar_small', 'ar_medium', 'ar_large',
)


def make_summary(status='PASS'):
    records = []
    for index, name in enumerate(summarize.EXPERIMENTS, 1):
        is_hrnet = name.startswith('rtdetr_hrnetv2')
        baseline = 0.70 if is_hrnet else 0.60
        if name.endswith('_pcx'):
            bonus = 0.01
        elif name.endswith('_esdr'):
            bonus = 0.02
        elif name.endswith('_psca'):
            bonus = 0.03
        else:
            bonus = 0.0
        score = baseline + bonus
        records.append({
            'index': index,
            'experiment': name,
            'status': status,
            'map_50_95': score,
            'map50': score + 0.20,
            'map75': score - 0.05,
            'ap_small': score - 0.10,
            'ap_medium': score - 0.04,
            'ap_large': score + 0.02,
            'ar_1': score - 0.20,
            'ar_10': score - 0.08,
            'ar_100': score - 0.03,
            'ar_small': score - 0.12,
            'ar_medium': score - 0.05,
            'ar_large': score + 0.01,
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


class UnifiedComparisonTests(unittest.TestCase):
    def test_metric_contract_contains_exactly_twelve_coco_metrics(self):
        self.assertEqual(len(summarize.METRICS), 12)
        self.assertEqual(
            tuple(field for _, field in summarize.METRICS), METRIC_FIELDS)
        self.assertEqual(
            tuple(public for public, _ in summarize.METRICS),
            ('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL',
             'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL'))
        self.assertEqual(summarize.GAIN_METRICS, ('AP', 'AP75', 'APS', 'ARS'))

    def test_protocol_and_eight_experiment_order_are_fixed(self):
        payload = summarize.build_comparison(
            make_summary(), Path('/tmp/summary.json'))
        self.assertEqual(
            [row['experiment'] for row in payload['records']],
            list(summarize.EXPERIMENTS))
        self.assertEqual(payload['protocol'], {
            'checkpoint': 'best.pth',
            'split': 'test',
            'gpu': '1',
            'num_workers': 2,
            'require_ema': True,
            'precision': 'FP32',
            'input_size': [640, 640],
        })
        self.assertTrue(payload['all_pass'])

    def test_gains_use_the_matching_backbone_baseline(self):
        payload = summarize.build_comparison(
            make_summary(), Path('/tmp/summary.json'))
        rows = {row['experiment']: row for row in payload['records']}

        pres_pcx = rows['rtdetr_r18vd_dut_anti_uav_pcx']
        self.assertEqual(
            pres_pcx['baseline_experiment'], summarize.PRES_BASELINE)
        self.assertAlmostEqual(pres_pcx['gains_vs_baseline']['AP'], 0.01)
        self.assertAlmostEqual(
            pres_pcx['gains_vs_baseline_percentage_points']['AP75'], 1.0)

        hr_psca = rows['rtdetr_hrnetv2_w18_dut_anti_uav_psca']
        self.assertEqual(
            hr_psca['baseline_experiment'], summarize.HR_BASELINE)
        self.assertAlmostEqual(hr_psca['gains_vs_baseline']['APS'], 0.03)
        self.assertAlmostEqual(
            hr_psca['gains_vs_baseline_percentage_points']['ARS'], 3.0)

        for baseline in (summarize.PRES_BASELINE, summarize.HR_BASELINE):
            for metric in summarize.GAIN_METRICS:
                self.assertEqual(
                    rows[baseline]['gains_vs_baseline'][metric], 0.0)

    def test_all_six_variants_use_their_own_family_baseline(self):
        payload = summarize.build_comparison(
            make_summary(), Path('/tmp/summary.json'))
        for row in payload['records']:
            expected = (summarize.HR_BASELINE
                        if row['family'] == 'HRNetV2-W18'
                        else summarize.PRES_BASELINE)
            self.assertEqual(row['baseline_experiment'], expected)

    def test_generate_writes_complete_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / 'summary.json'
            summary_path.write_text(
                json.dumps(make_summary()), encoding='utf-8')

            payload, json_path, markdown_path = summarize.generate(
                summary_path, root / 'reports')

            self.assertTrue(payload['all_pass'])
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            saved = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(len(saved['records']), 8)
            self.assertEqual(len(saved['records'][0]['metrics']), 12)
            markdown = markdown_path.read_text(encoding='utf-8')
            for heading in (
                    'AP', 'AP50', 'AP75', 'APS', 'APM', 'APL',
                    'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL'):
                self.assertIn(heading, markdown)
            self.assertIn(
                'Gains relative to the matching backbone baseline', markdown)
            self.assertIn('| PResNet18 | PCX |', markdown)
            self.assertIn('| HRNetV2-W18 | PSCA |', markdown)

    def test_missing_metric_writes_incomplete_report_and_returns_nonzero(self):
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
            payload = json.loads((
                root / 'out' / summarize.DEFAULT_JSON_NAME
            ).read_text(encoding='utf-8'))
            self.assertFalse(payload['all_pass'])
            markdown = (
                root / 'out' / summarize.DEFAULT_MARKDOWN_NAME
            ).read_text(encoding='utf-8')
            self.assertIn('Incomplete evaluations', markdown)
            self.assertIn('missing metrics: ARS', markdown)

    def test_failed_record_has_no_gains_and_returns_nonzero(self):
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
            payload = json.loads((
                root / 'out' / summarize.DEFAULT_JSON_NAME
            ).read_text(encoding='utf-8'))
            failed = payload['records'][2]
            self.assertTrue(all(
                value is None
                for value in failed['gains_vs_baseline'].values()))

    def test_wrong_protocol_is_rejected(self):
        mutations = (
            ('split', 'val'),
            ('gpu', '2'),
            ('num_workers', 4),
            ('require_ema', False),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                data = make_summary()
                data[key] = value
                with self.assertRaisesRegex(ValueError, 'fixed test protocol'):
                    summarize.build_comparison(
                        data, Path('/tmp/summary.json'))

    def test_missing_or_wrong_experiment_order_is_rejected(self):
        missing = make_summary()
        missing['records'].pop()
        with self.assertRaisesRegex(ValueError, 'order/set mismatch'):
            summarize.build_comparison(
                missing, Path('/tmp/summary.json'))

        wrong_order = make_summary()
        wrong_order['records'][0], wrong_order['records'][1] = (
            wrong_order['records'][1], wrong_order['records'][0])
        with self.assertRaisesRegex(ValueError, 'order/set mismatch'):
            summarize.build_comparison(
                wrong_order, Path('/tmp/summary.json'))


class UnifiedTestShellContractTests(unittest.TestCase):
    def test_wrapper_has_exact_order_and_fixed_test_protocol(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / 'tools' / 'test_pcx_esdr_psca_best.sh')
        script = script_path.read_text(encoding='utf-8')
        match = __import__('re').search(
            r'(?ms)^EXPERIMENTS=\(\s*(.*?)^\)', script)
        self.assertIsNotNone(match)
        experiments = tuple(__import__('re').findall(
            r'^\s*"([^"]+)"\s*$', match.group(1), __import__('re').M))
        self.assertEqual(experiments, summarize.EXPERIMENTS)

        self.assertIn('GPU_ID=1', script)
        self.assertIn('NUM_WORKERS=2', script)
        self.assertIn('--split test', script)
        self.assertIn('--require-ema', script)
        self.assertIn('tools/test_all_best.py', script)
        self.assertIn('tools/summarize_pcx_esdr_psca.py', script)
        self.assertNotIn('--amp', script)
        self.assertIn('test_status=$?', script)
        self.assertIn('summary_status', script)
        self.assertIn('best.pth', summarize.render_markdown(
            summarize.build_comparison(
                make_summary(), Path('/tmp/summary.json'))))


if __name__ == '__main__':
    unittest.main()
