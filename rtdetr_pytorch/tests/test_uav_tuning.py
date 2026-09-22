"""Config isolation, dataset-stat and training-log checks for UAV tuning."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_dut_models import differences, fresh_config, import_model_source
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

    def test_augmented_configs_change_only_augmentation(self):
        baseline = fresh_config(CONFIG_ROOT / 'rtdetr_r18vd_dut_anti_uav.yml')
        names = ('A0_coco_style', 'A1_no_zoom_crop08', 'A2_no_zoom_crop04',
                 'A3_no_zoom_crop02', 'A4_no_zoom_no_crop')
        expected_p = (0.8, 0.8, 0.4, 0.2, None)
        for name, probability in zip(names, expected_p):
            candidate = fresh_config(TUNING / (name + '.yml'))
            keys = set(differences(baseline, candidate))
            allowed = {'__include__', 'output_dir',
                       'train_dataloader.dataset.transforms.ops'}
            self.assertLessEqual(keys, allowed, name)
            ops = candidate['train_dataloader']['dataset']['transforms']['ops']
            types = [op['type'] for op in ops]
            self.assertEqual('RandomZoomOut' in types, name == names[0])
            crop = [op for op in ops if op['type'] == 'RandomIoUCrop']
            self.assertEqual(crop[0]['p'] if crop else None, probability)
            self.assertEqual(candidate['train_dataloader']['batch_size'], 16)
            self.assertEqual(candidate['epoches'], 200)
            self.assertFalse(candidate['MERT']['enabled'])
            self.assertEqual(candidate['BackboneEnhancement'],
                             {'bafr': False, 'hcbr': False})

    def test_resolution_eval_cache_matches_validation_resize(self):
        for name, size in (('B0_a2_640', 640),
                           ('uav_baseline_v1', 640),
                           ('B1_a2_800', 800),
                           ('B2_a2_960', 960)):
            config = fresh_config(TUNING / (name + '.yml'))
            for loader in ('train_dataloader', 'val_dataloader'):
                ops = config[loader]['dataset']['transforms']['ops']
                self.assertEqual([op['size'] for op in ops if op['type'] == 'Resize'],
                                 [[size, size]])
            self.assertEqual(config['HybridEncoder']['eval_spatial_size'], [size, size])
            self.assertEqual(config['RTDETRTransformer']['eval_spatial_size'], [size, size])
            self.assertEqual(config['RTDETRTransformer']['num_queries'], 300)
            self.assertEqual(config['RTDETRTransformer']['num_denoising'], 100)
            self.assertEqual(config['SetCriterion']['weight_dict'],
                             {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2})
        core = import_model_source(selective=True)
        import torch
        torch.set_num_threads(2)
        with torch.no_grad():
            for name, size in (('uav_baseline_v1', 640),
                               ('B1_a2_800', 800), ('B2_a2_960', 960)):
                model = core.YAMLConfig(str(TUNING / (name + '.yml')),
                                        PResNet={'pretrained': False}).model.eval()
                output = model(torch.randn(1, 3, size, size))
                self.assertEqual(output['pred_logits'].shape, (1, 300, 1))
                self.assertEqual(output['pred_boxes'].shape, (1, 300, 4))

    def test_learning_rate_candidate_only_changes_backbone_groups(self):
        base = fresh_config(TUNING / 'uav_baseline_v1.yml')
        control = fresh_config(TUNING / 'C0_a2_640_backbone1e5.yml')
        candidate = fresh_config(TUNING / 'C1_a2_640_backbone2e5.yml')
        self.assertEqual(control['optimizer'], base['optimizer'])
        keys = set(differences(base, candidate))
        self.assertLessEqual(keys, {'__include__', 'output_dir', 'optimizer.params'})
        self.assertEqual([group['lr'] for group in candidate['optimizer']['params'][:2]],
                         [2e-5, 2e-5])
        self.assertEqual(candidate['optimizer']['lr'], base['optimizer']['lr'])

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
