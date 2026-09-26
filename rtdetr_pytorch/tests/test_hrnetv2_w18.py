"""Acceptance tests for the independent HRNetV2-W18 backbone experiment."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests._support import PROJECT_DIR, prepare_imports
prepare_imports()

from src.core import YAMLConfig  # noqa: E402
import src.nn.backbone.hrnet as hrnet_module  # noqa: E402
from src.nn.backbone.hrnet import HRNetV2W18  # noqa: E402
from src.nn.backbone.presnet import PResNet  # noqa: E402
from tools.analyze_dut_models import differences, fresh_config  # noqa: E402
from tools.resolve_config_output import resolve_output_dir  # noqa: E402


ROOT = PROJECT_DIR


BASELINE = ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml'
HRNET = ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml'


def build_hrnet_detector():
    config = YAMLConfig(
        str(HRNET),
        HRNetV2W18={'pretrained': False, 'pretrained_path': None})
    model = config.model.eval()
    model.multi_scale = None
    return config, model


class HRNetV2W18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_standard_stage_layout_and_640_shapes(self):
        model = HRNetV2W18(pretrained=False).eval()
        self.assertEqual([len(model.stage2), len(model.stage3), len(model.stage4)],
                         [1, 4, 3])
        self.assertEqual([module.num_branches for module in model.stage2], [2])
        self.assertEqual([module.num_branches for module in model.stage3], [3] * 4)
        self.assertEqual([module.num_branches for module in model.stage4], [4] * 3)
        with torch.inference_mode():
            outputs = model(torch.randn(1, 3, 640, 640))
        self.assertEqual([tuple(value.shape) for value in outputs], [
            (1, 36, 80, 80), (1, 72, 40, 40), (1, 144, 20, 20)])
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs))

    def test_all_stages_receive_gradients(self):
        torch.manual_seed(3)
        model = HRNetV2W18(pretrained=False).train()
        outputs = model(torch.randn(1, 3, 128, 128))
        loss = sum((value * torch.randn_like(value)).mean() for value in outputs)
        loss.backward()
        representatives = {
            'stem': model.conv1.weight,
            'stage1': model.layer1[0].conv1.weight,
            'stage2': model.stage2[0].branches[0][0].conv1.weight,
            'stage3': model.stage3[0].branches[0][0].conv1.weight,
            'stage4': model.stage4[0].branches[0][0].conv1.weight,
        }
        for name, parameter in representatives.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)
        missing = [name for name, parameter in model.named_parameters()
                   if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(missing, [], 'DDP would see unused parameters')

    def test_full_detector_forward_is_finite(self):
        _, model = build_hrnet_detector()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 640, 640))
        self.assertEqual(tuple(output['pred_logits'].shape), (1, 300, 1))
        self.assertEqual(tuple(output['pred_boxes'].shape), (1, 300, 4))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()
                            if torch.is_tensor(value)))

    def test_full_detector_backward_reaches_every_backbone_parameter(self):
        _, model = build_hrnet_detector()
        model.train()
        targets = [{'labels': torch.tensor([0]),
                    'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]])}]
        output = model(torch.randn(1, 3, 128, 128), targets)
        loss = (output['pred_logits'].square().mean()
                + output['pred_boxes'].square().mean())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        missing = [name for name, parameter in model.backbone.named_parameters()
                   if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(missing, [])

    def test_yaml_changes_only_backbone_contract_and_output(self):
        baseline = fresh_config(BASELINE)
        candidate = fresh_config(HRNET)
        diff = differences(baseline, candidate)
        allowed = {
            '__include__', 'output_dir', 'RTDETR.backbone',
            'HRNetV2W18.pretrained', 'HRNetV2W18.pretrained_path',
            'HybridEncoder.in_channels',
        }
        self.assertEqual(set(diff), allowed)
        self.assertEqual(candidate['HybridEncoder']['feat_strides'], [8, 16, 32])
        for key in ('optimizer', 'lr_scheduler', 'epoches', 'checkpoint_step',
                    'train_dataloader', 'val_dataloader', 'RTDETRTransformer',
                    'SetCriterion', 'MERT', 'SECD', 'PDR'):
            self.assertEqual(candidate.get(key), baseline.get(key), key)

    def test_output_resolver_honors_yaml_and_final_cli_override(self):
        self.assertEqual(
            resolve_output_dir(HRNET),
            './output/rtdetr_hrnetv2_w18_dut_anti_uav')
        self.assertEqual(
            resolve_output_dir(HRNET, 'output/final-cli-path'),
            'output/final-cli-path')

    def test_classification_head_is_the_only_permitted_pretrained_extra(self):
        reference = HRNetV2W18(pretrained=False)
        state = copy.deepcopy(reference.state_dict())
        state['classifier.weight'] = torch.randn(1000, 2048)
        state['classifier.bias'] = torch.randn(1000)
        with patch('torch.hub.load_state_dict_from_url', return_value=state):
            loaded = HRNetV2W18(pretrained=True)
        report = loaded.pretrained_load_report
        self.assertEqual(len(report['matched_keys']), len(reference.state_dict()))
        self.assertEqual(report['missing_keys'], [])
        self.assertEqual(report['unexpected_keys'], [])
        self.assertEqual(report['ignored_classifier_keys'],
                         ['classifier.bias', 'classifier.weight'])

    def test_backbone_mismatch_fails_loudly(self):
        state = copy.deepcopy(HRNetV2W18(pretrained=False).state_dict())
        state.pop('conv1.weight')
        with patch('torch.hub.load_state_dict_from_url', return_value=state):
            with self.assertRaisesRegex(RuntimeError, 'missing=.*conv1.weight'):
                HRNetV2W18(pretrained=True)

    def test_project_local_weight_is_preferred_without_network(self):
        state = copy.deepcopy(HRNetV2W18(pretrained=False).state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hrnetv2_w18-8cb57bb9.pth'
            torch.save(state, path)
            with patch.object(hrnet_module, '_PROJECT_WEIGHT_PATH', path), \
                    patch('torch.hub.load_state_dict_from_url') as download:
                loaded = HRNetV2W18(pretrained=True)
        download.assert_not_called()
        self.assertEqual(loaded.pretrained_load_report['source'], str(path))

    def test_ddp_rank_zero_is_the_only_downloader(self):
        state = copy.deepcopy(HRNetV2W18(pretrained=False).state_dict())
        with patch.object(hrnet_module, '_PROJECT_WEIGHT_PATH',
                          Path('does-not-exist.pth')), \
                patch.object(hrnet_module.torch_dist, 'is_available',
                             return_value=True), \
                patch.object(hrnet_module.torch_dist, 'is_initialized',
                             return_value=True), \
                patch.object(hrnet_module.torch_dist, 'get_world_size',
                             return_value=3), \
                patch.object(hrnet_module.torch_dist, 'get_rank',
                             return_value=0), \
                patch.object(hrnet_module.torch_dist, 'broadcast_object_list') \
                as broadcast, \
                patch('torch.hub.load_state_dict_from_url', return_value=state) \
                as download:
            HRNetV2W18(pretrained=True)
        download.assert_called_once()
        broadcast.assert_called_once()

    def test_download_error_exposes_root_cause_and_expected_paths(self):
        with patch.object(hrnet_module, '_PROJECT_WEIGHT_PATH',
                          Path('does-not-exist.pth')), \
                patch('torch.hub.load_state_dict_from_url',
                      side_effect=OSError('network unreachable')):
            with self.assertRaisesRegex(
                    RuntimeError, 'Underlying error: OSError: network unreachable'):
                HRNetV2W18(pretrained=True)

    def test_building_hrnet_does_not_pollute_baseline(self):
        _, candidate = build_hrnet_detector()
        self.assertIsInstance(candidate.backbone, HRNetV2W18)
        baseline_config = YAMLConfig(
            str(BASELINE), PResNet={'pretrained': False})
        baseline = baseline_config.model
        self.assertIsInstance(baseline.backbone, PResNet)
        self.assertEqual(baseline_config.yaml_cfg['HybridEncoder']['in_channels'],
                         [128, 256, 512])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for AMP')
    def test_cuda_amp_forward_backward(self):
        model = HRNetV2W18(pretrained=False).cuda().train()
        image = torch.randn(1, 3, 256, 256, device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(image)
            loss = sum(value.square().mean() for value in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.conv1.weight.grad)


if __name__ == '__main__':
    unittest.main()
