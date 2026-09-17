"""Original-protocol and solver integration tests with synthetic inputs only."""

import ast
import copy
import io
import json
import runpy
import subprocess
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from tests._support import PROJECT_DIR, prepare_imports
prepare_imports()

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.nn.backbone import presnet
from src.solver import det_engine
from src.solver.mert import MERT, decoder_box_trajectory


PREFIX = 'rtdetr_r18vd_dut_anti_uav'


def original_source(path):
    return subprocess.check_output(
        ['git', 'show', 'HEAD:rtdetr_pytorch/' + path], cwd=PROJECT_DIR,
        text=True, encoding='utf-8',
    )


def original_namespace(path, package):
    source = original_source(path)
    source = source.replace('from src.core import register', 'register = lambda cls: cls')
    namespace = {'__name__': '_original_reference', '__package__': package}
    exec(compile(source, path, 'exec'), namespace)
    return namespace


class DUTIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.original_presnet = original_namespace('src/nn/backbone/presnet.py',
                                                  'src.nn.backbone')['PResNet']
        cls.original_engine = original_namespace('src/solver/det_engine.py', 'src.solver')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def model(self, suffix='', original=False):
        cfg = YAMLConfig(str(PROJECT_DIR / 'configs/rtdetr' / (PREFIX + suffix + '.yml')))
        cfg.yaml_cfg['PResNet']['pretrained'] = False  # TEST ONLY: no network access
        torch.manual_seed(17)
        if original:
            with mock.patch.object(presnet, 'PResNet', self.original_presnet):
                model = cfg.model
        else:
            model = cfg.model
        return cfg, model

    def test_official_yaml_files_unmodified(self):
        paths = subprocess.check_output(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', 'rtdetr_pytorch/configs'],
            cwd=PROJECT_DIR.parent, text=True, encoding='utf-8',
        ).splitlines()
        self.assertTrue(paths)
        for path in paths:
            # DUT configs are intentionally editable experiment adapters, not
            # official YAML; they may have been committed after initial setup.
            if 'dut_anti_uav' in path:
                continue
            with self.subTest(path=path):
                relative = path[len('rtdetr_pytorch/'):]
                self.assertEqual((PROJECT_DIR / relative).read_text(encoding='utf-8'),
                                 original_source(relative))

    def test_baseline_matches_original_full_model_keys_params_shapes_and_values(self):
        _, reference = self.model(original=True)
        _, current = self.model()
        self.assertEqual(set(reference.state_dict()), set(current.state_dict()))
        self.assertEqual(sum(p.numel() for p in reference.parameters()),
                         sum(p.numel() for p in current.parameters()))
        for key, tensor in reference.state_dict().items():
            self.assertTrue(torch.equal(tensor, current.state_dict()[key]), key)
        images = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            original_outputs = reference.eval()(images)
            outputs = current.eval()(images)
        for key in outputs:
            self.assertTrue(torch.equal(outputs[key], original_outputs[key]), key)
        self.assertEqual(tuple(outputs['pred_boxes'].shape), (1, 300, 4))
        self.assertEqual(tuple(outputs['pred_logits'].shape), (1, 300, 1))

    def test_secd_alpha_zero_full_model_preserves_original_weights_and_outputs(self):
        _, baseline = self.model()
        baseline.eval()
        images = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            expected = baseline(images)
        for suffix in ('_secd_34', '_secd_45', '_secd_345'):
            with self.subTest(method=suffix):
                _, model = self.model(suffix)
                for key, tensor in baseline.state_dict().items():
                    self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
                added = set(model.state_dict()) - set(baseline.state_dict())
                self.assertTrue(added)
                self.assertTrue(all(key.startswith(('backbone.secd_34.', 'backbone.secd_45.'))
                                    for key in added))
                with torch.no_grad():
                    actual = model.eval()(images)
                for key in expected:
                    self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_original_coco_loaded_after_methods_does_not_keep_secd_or_dut_classes(self):
        self.model('_secd_345_mert_late_xywh')
        cfg = YAMLConfig(str(PROJECT_DIR / 'configs/rtdetr/rtdetr_r18vd_6x_coco.yml'))
        cfg.yaml_cfg['PResNet']['pretrained'] = False
        self.assertNotIn('MERT', cfg.yaml_cfg)
        self.assertEqual(cfg.yaml_cfg['num_classes'], 80)
        model = cfg.model
        self.assertIsNone(model.backbone.secd_34)
        self.assertIsNone(model.backbone.secd_45)
        self.assertEqual(model.decoder.num_classes, 80)

    def test_train_baseline_update_identical_to_original_and_no_mert_created(self):
        torch.manual_seed(1)
        baseline = _ToyDetector()
        modified = copy.deepcopy(baseline)
        images = torch.randn(2, 3, 8, 8)
        targets = [{'boxes': torch.full((1, 4), .5)} for _ in range(2)]
        loader = [(images, targets)]
        with redirect_stdout(io.StringIO()):
            self.original_engine['train_one_epoch'](
                baseline, _ToyCriterion(), loader,
                torch.optim.SGD(baseline.parameters(), lr=.01), torch.device('cpu'), 0,
            )
            with mock.patch.object(det_engine, 'MERT', side_effect=AssertionError('Disabled plugin constructed')):
                det_engine.train_one_epoch(
                    modified, _ToyCriterion(), loader,
                    torch.optim.SGD(modified.parameters(), lr=.01), torch.device('cpu'), 0,
                    mert_config={'enabled': False},
                )
        self.assertEqual(baseline.calls, modified.calls)
        self.assertEqual(modified.calls, 1)
        for key, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(tensor, modified.state_dict()[key]), key)

    def test_secd_mert_concat_native_criterion_dn_and_backward(self):
        cfg, model = self.model('_secd_34_mert_late_xywh')
        model.train()
        model.multi_scale = None  # TEST ONLY: synthetic small forward for speed
        images = torch.randn(1, 3, 128, 128)
        targets = [_target()]
        plugin = MERT(cfg.yaml_cfg['MERT'])
        pair = plugin.prepare(images, targets, model, shifts=[[1, -1]])
        calls = []
        handle = model.register_forward_hook(lambda *_: calls.append(1))
        outputs = det_engine._forward_mert_views(model, pair, plugin)
        handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertIn('dn_aux_outputs', outputs)
        self.assertEqual(tuple(decoder_box_trajectory(outputs).shape), (3, 2, 300, 4))
        losses = det_engine._mert_train_losses(outputs, pair, plugin, cfg.criterion)
        self.assertIn('loss_mert', losses)
        self.assertTrue(any('_dn_' in name for name in losses))
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(model.backbone.secd_34.raw_alpha.grad)
        self.assertTrue(torch.isfinite(model.backbone.secd_34.raw_alpha.grad))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))
        self.assertIsNone(model.multi_scale)

    def test_evaluate_is_exact_original_code_and_contains_no_new_amp_patch(self):
        original = ast.parse(original_source('src/solver/det_engine.py'))
        current = ast.parse((PROJECT_DIR / 'src/solver/det_engine.py').read_text(encoding='utf-8'))
        def evaluate_node(tree):
            return next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'evaluate')
        self.assertEqual(ast.dump(evaluate_node(original)), ast.dump(evaluate_node(current)))
        self.assertEqual((PROJECT_DIR / 'tools/train.py').read_text(encoding='utf-8'),
                         original_source('tools/train.py'))

    def test_native_evaluate_calls_model_and_secd_once_and_never_mert(self):
        cfg, model = self.model('_secd_34_mert_late_xywh')
        counts = {}
        def count(name):
            def hook(*args):
                counts[name] = counts.get(name, 0) + 1
            return hook
        handles = [child.register_forward_hook(count(name)) for name, child in (
            ('model', model), ('backbone', model.backbone),
            ('encoder', model.encoder), ('decoder', model.decoder),
            ('secd_34', model.backbone.secd_34),
            ('secd_34_evidence', model.backbone.secd_34.projection),
        )]
        evaluator = types.SimpleNamespace(
            coco_eval={'bbox': types.SimpleNamespace(stats=np.zeros(12))},
            update=mock.Mock(), synchronize_between_processes=mock.Mock(),
            accumulate=mock.Mock(), summarize=mock.Mock(),
        )
        loader = [(torch.zeros(1, 3, 640, 640), [{
            'image_id': torch.tensor([1]), 'orig_size': torch.tensor([720, 1200]),
        }])]
        try:
            with mock.patch.object(det_engine, 'CocoEvaluator', return_value=evaluator), \
                    mock.patch.object(MERT, 'prepare', side_effect=AssertionError('MERT in eval')), \
                    redirect_stdout(io.StringIO()):
                det_engine.evaluate(model, nn.Identity(), cfg.postprocessor, loader,
                                    None, torch.device('cpu'), None)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts, {name: 1 for name in (
            'model', 'backbone', 'encoder', 'decoder', 'secd_34', 'secd_34_evidence')})

    def test_real_test_split_adapter_changes_only_dataset_paths(self):
        cfg = YAMLConfig(str(PROJECT_DIR / 'configs/rtdetr' / (PREFIX + '.yml')))
        before = copy.deepcopy(cfg.yaml_cfg)
        select_split = runpy.run_path(str(PROJECT_DIR / 'tools/test_dut.py'))['select_split']
        paths = select_split(cfg, 'test')
        self.assertTrue(paths['ann_file'].endswith('/test.json'))
        self.assertTrue(paths['img_folder'].endswith('/images/test/'))
        before['val_dataloader']['dataset'].update(before['test_dataset'])
        self.assertEqual(before, cfg.yaml_cfg)
        self.assertEqual(cfg.yaml_cfg['val_dataloader']['batch_size'], 16)

    def test_data_categories_are_read_and_consistent_with_one_zero_based_class(self):
        root = PROJECT_DIR.parent / 'DUT-Anti-UAV/DUT-Anti-UAV'
        if not root.is_dir():
            self.skipTest('Actual DUT data are not present on this checkout')
        for split in ('train', 'val', 'test'):
            data = json.loads((root / 'labels' / (split + '.json')).read_text(encoding='utf-8'))
            self.assertEqual([cat['id'] for cat in data['categories']], [0])
            self.assertTrue(all(annotation['category_id'] == 0 for annotation in data['annotations']))


def _target():
    return {'boxes': torch.tensor([[.5, .5, .03, .04]]), 'labels': torch.tensor([0]),
            'area': torch.tensor([19.66]), 'iscrowd': torch.tensor([0]),
            'image_id': torch.tensor([1]), 'size': torch.tensor([128, 128]),
            'orig_size': torch.tensor([128, 128])}


class _ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 5)
        self.calls = 0

    def forward(self, images, targets=None):
        self.calls += 1
        output = self.backbone(images.mean(dim=(-2, -1)))[:, None]
        return {'pred_logits': output[..., :1], 'pred_boxes': output[..., 1:].sigmoid()}


class _ToyCriterion(nn.Module):
    def forward(self, outputs, targets):
        expected = torch.stack([target['boxes'] for target in targets])
        return {'loss_det': (outputs['pred_boxes'] - expected).square().mean()
                + outputs['pred_logits'].square().mean()}


if __name__ == '__main__':
    unittest.main()
