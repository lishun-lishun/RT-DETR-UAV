"""Config isolation, dataset-stat and training-log checks for UAV tuning."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path

from tools.analyze_dut_models import fresh_config
from tools.analyze_training_log import analyze as analyze_log
from tools.analyze_uav_dataset import analyze as analyze_dataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'configs' / 'rtdetr'
TUNING = CONFIG_ROOT / 'uav_tuning'


class UAVTuningTests(unittest.TestCase):
    def test_dataset_statistics_and_direct_square_resize(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            images = [{'id': 1, 'width': 200, 'height': 100},
                      {'id': 2, 'width': 100, 'height': 100}]
            train = {'images': images, 'categories': [{'id': 0, 'name': 'UAV'}],
                     'annotations': [
                         {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 2, 2]},
                         {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 4, 4]},
                         {'image_id': 2, 'category_id': 0, 'bbox': [0, 0, 1, 1]},
                     ]}
            val = {'images': [images[0]], 'categories': train['categories'],
                   'annotations': [train['annotations'][0]]}
            train_path, val_path = folder / 'train.json', folder / 'val.json'
            train_path.write_text(json.dumps(train), encoding='utf-8')
            val_path.write_text(json.dumps(val), encoding='utf-8')
            report = analyze_dataset(train_path, val_path)
            self.assertEqual(report['train']['images'], 2)
            self.assertEqual(report['val']['images'], 1)
            self.assertEqual(report['train']['objects_per_image']['max'], 2)
            self.assertEqual(report['bbox_train']['width']['median'], 2)
            self.assertEqual(report['resized_train']['640']['both_below_px']['8']['count'], 1)
            self.assertEqual(report['small_original_train']['both_below_px']['4']['count'], 2)

    def test_three_gpu_learning_rate_configs(self):
        baseline = fresh_config(CONFIG_ROOT / 'rtdetr_r18vd_dut_anti_uav.yml')
        candidates = {
            'G48_lr1x': (1e-4, 1e-5),
            'G48_lr2x': (2e-4, 2e-5),
            'G48_lr3x': (3e-4, 3e-5),
        }
        baseline_ops = baseline['train_dataloader']['dataset']['transforms']['ops']
        for name, (main_lr, backbone_lr) in candidates.items():
            candidate = fresh_config(TUNING / (name + '.yml'))
            self.assertEqual(candidate['expected_world_size'], 3)
            self.assertTrue(candidate['plot_training_curves'])
            self.assertEqual(candidate['train_dataloader']['batch_size'], 16)
            self.assertEqual(candidate['val_dataloader']['batch_size'], 16)
            self.assertEqual(candidate['val_dataloader']['num_workers'], 2)
            self.assertEqual(candidate['epoches'], 200)
            self.assertEqual(candidate['checkpoint_step'], 10)
            self.assertEqual(candidate['ema']['warmups'], 667)
            self.assertEqual(candidate['optimizer']['lr'], main_lr)
            self.assertEqual([group['lr'] for group in
                              candidate['optimizer']['params'][:2]],
                             [backbone_lr, backbone_lr])
            self.assertEqual(
                candidate['train_dataloader']['dataset']['transforms']['ops'],
                baseline_ops)
            self.assertFalse(candidate['MERT']['enabled'])
            self.assertEqual(candidate['BackboneEnhancement'],
                             {'bafr': False, 'hcbr': False})

    def test_combined_curve_png(self):
        plotter = runpy.run_path(str(ROOT / 'src' / 'solver' / 'training_plot.py'))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            log_path = folder / 'log.txt'
            output_path = folder / 'training_curves.png'
            entries = [
                {'epoch': epoch, 'train_loss': 10 - epoch,
                 'test_coco_eval_bbox': [0.4 + epoch * .01,
                                         0.7 + epoch * .01,
                                         0.45 + epoch * .01] + [-1] * 9}
                for epoch in range(3)
            ]
            log_path.write_text(''.join(json.dumps(item) + '\n' for item in entries),
                                encoding='utf-8')
            self.assertTrue(plotter['plot_training_curves'](log_path, output_path))
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 1000)

    def test_log_parser_best_epochs_and_trend(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'log.txt'
            entries = []
            for epoch in range(12):
                ap = 0.60 if epoch == 1 else 0.55 - 0.002 * epoch
                scores = [ap, 0.8 - 0.001 * epoch, 0.4 - 0.001 * epoch,
                          0.3 - 0.001 * epoch] + [-1] * 8
                entries.append({'epoch': epoch, 'train_lr': 1e-4,
                                'train_loss': 12 - epoch,
                                'train_loss_bbox': 2 - epoch * .01,
                                'train_loss_giou': 3 - epoch * .01,
                                'train_loss_vfl': 1 - epoch * .01,
                                'test_coco_eval_bbox': scores})
            path.write_text(''.join(json.dumps(item) + '\n' for item in entries),
                            encoding='utf-8')
            report = analyze_log(path)
            self.assertEqual(report['best']['AP']['completed_epoch'], 2)
            self.assertEqual(report['best']['AP50']['completed_epoch'], 1)
            self.assertEqual(report['trend']['assessment'],
                             'possible_overfitting_or_late_training_instability')


if __name__ == '__main__':
    unittest.main()
