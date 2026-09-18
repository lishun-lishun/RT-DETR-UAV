"""Training/save orchestration tests; no Torch/GPU/dataset dependency.

Compile the actual solver class bodies unchanged. Explicit test doubles model
training/evaluation/serialization, NOT real detection or AMP/model accuracy.
Run: python -m unittest discover -s tests -p test_training_checkpoints.py -v
"""

import ast
import copy
import datetime
import io
import json
import math
from pathlib import Path
import pickle
import runpy
import tempfile
import time
import types
from typing import Dict
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def solver_classes():
    namespace = {
        'BaseConfig': object, 'Dict': Dict, 'nn': types.SimpleNamespace(Module=object),
        'torch': types.SimpleNamespace(Tensor=object), 'datetime': datetime.datetime,
        'Path': Path,
    }
    for filename in ('solver.py', 'det_solver.py'):
        source = ast.parse((ROOT / 'src/solver' / filename).read_text(encoding='utf-8'))
        classes = [node for node in source.body if isinstance(node, ast.ClassDef)]
        if filename == 'det_solver.py':
            namespace.update(datetime=types.SimpleNamespace(
                now=datetime.datetime.now, timedelta=datetime.timedelta),
                time=time, json=json, math=math)
        exec(compile(ast.Module(body=classes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class FakeState:
    def __init__(self, value=-1):
        self.value = value

    def state_dict(self):
        return {'value': self.value}

    def load_state_dict(self, state):
        self.value = state['value']

    def step(self):
        self.value += 1


class FakeModel(FakeState):
    def parameters(self):
        return [types.SimpleNamespace(requires_grad=True, numel=lambda: 1)]


class FakeEMA(FakeState):
    def __init__(self):
        super().__init__()
        self.module = FakeModel()

    def state_dict(self):
        return {'module': self.module.state_dict()}


class TrainingCheckpointTests(unittest.TestCase):
    def make_solver(self, directory, aps, main=True, ema=False, resume=None, step=10):
        namespace = solver_classes()
        cfg = types.SimpleNamespace(epoches=len(aps), checkpoint_step=step,
                                    clip_max_norm=.1, log_step=100, yaml_cfg={})
        solver = namespace['DetSolver'](cfg)
        solver.model = FakeModel()
        solver.optimizer = FakeState()
        solver.lr_scheduler = FakeState()
        solver.scaler = FakeState()
        solver.criterion = object()
        solver.postprocessor = object()
        solver.device = 'test-only'
        solver.ema = FakeEMA() if ema else None
        solver.output_dir = Path(directory)
        solver.last_epoch = -1
        loader = types.SimpleNamespace(dataset=object(), sampler=types.SimpleNamespace(set_epoch=Mock()))
        solver.train_dataloader = solver.val_dataloader = loader
        solver.train = lambda: solver.load_state_dict(resume) if resume is not None else None
        writes, timeline = [], []

        def save(state, path):
            timeline.append(('save', state['last_epoch']))
            writes.append((Path(path).name, copy.deepcopy(state), id(state)))
            with Path(path).open('wb') as stream:
                pickle.dump(state, stream)

        def train(model, criterion, data, optimizer, device, epoch, *args, **kwargs):
            timeline.append(('train', epoch))
            model.value = epoch
            if solver.ema is not None:
                solver.ema.module.value = epoch + 100
            return {'loss': 1.0}

        def evaluate(module, *args):
            epoch = solver.model.value
            timeline.append(('eval', epoch))
            self.assertIs(module, solver.ema.module if solver.ema else solver.model)
            # AP50 / mask AP deliberately disagree with bbox AP to ensure only
            # bbox AP@[.50:.95] selects the model, and no other metric its epoch.
            return {'coco_eval_bbox': [aps[epoch], .99 - epoch * .001],
                    'coco_eval_masks': [.99]}, None

        namespace.update(
            dist=types.SimpleNamespace(is_dist_available_and_initialized=lambda: False,
                                       is_main_process=lambda: main,
                                       save_on_master=save, de_parallel=lambda model: model,
                                       is_parallel=lambda model: False),
            get_coco_api_from_dataset=lambda dataset: object(),
            train_one_epoch=train, evaluate=evaluate,
        )
        real_state_dict = solver.state_dict
        solver.state_dict = Mock(wraps=real_state_dict)
        return solver, writes, timeline

    @staticmethod
    def read(directory, filename):
        with (Path(directory) / filename).open('rb') as stream:
            return pickle.load(stream)

    def fit(self, solver):
        with redirect_stdout(io.StringIO()):
            solver.fit()

    def test_every_dut_yaml_inherits_200_epochs_and_ten_epoch_saves(self):
        fresh_config = runpy.run_path(str(ROOT / 'tools/analyze_dut_models.py'))['fresh_config']
        paths = sorted((ROOT / 'configs/rtdetr').glob('*dut_anti_uav*.yml'))
        self.assertGreaterEqual(len(paths), 12)
        official = fresh_config(ROOT / 'configs/rtdetr/rtdetr_r18vd_6x_coco.yml')
        for path in paths:
            cfg = fresh_config(path)
            with self.subTest(config=path.name):
                self.assertEqual(cfg['epoches'], 200)
                self.assertEqual(cfg['checkpoint_step'], 10)
                self.assertEqual(cfg['optimizer'], official['optimizer'])
                self.assertEqual(cfg['lr_scheduler'], official['lr_scheduler'])

    def test_all_200_epochs_validate_and_twenty_numbered_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, writes, timeline = self.make_solver(directory, [.2] * 199 + [.3])
            self.fit(solver)
            self.assertEqual(sum(event[0] == 'eval' for event in timeline), 200)
            numbered = sorted(name for name, _, _ in writes if name.startswith('checkpoint0'))
            self.assertEqual(numbered, [f'checkpoint{epoch:04}.pth' for epoch in range(9, 200, 10)])
            self.assertEqual(solver.state_dict.call_count, 200)
            best = self.read(directory, 'best.pth')
            self.assertEqual(best['last_epoch'], 199)
            self.assertEqual(best['best_stat'], {'epoch': 199, 'coco_eval_bbox': .3})
            self.assertEqual(self.read(directory, 'checkpoint.pth')['last_epoch'], 199)
            rows = [json.loads(line) for line in (Path(directory) / 'log.txt').read_text().splitlines()]
            self.assertEqual(len(rows), 200)
            self.assertEqual(rows[-1]['best_stat'], best['best_stat'])

    def test_best_saves_immediately_off_period_and_never_regresses(self):
        aps = [.3] + [.2] * 24
        aps[9], aps[14] = .5, .6
        with tempfile.TemporaryDirectory() as directory:
            solver, writes, timeline = self.make_solver(directory, aps)
            self.fit(solver)
            best = self.read(directory, 'best.pth')
            self.assertEqual(best['last_epoch'], 14)
            self.assertEqual(best['model']['value'], 14)
            self.assertEqual(best['best_stat']['coco_eval_bbox'], .6)
            self.assertEqual(best['validation_stats']['coco_eval_bbox'][0], .6)
            self.assertEqual([state['last_epoch'] for name, state, _ in writes if name == 'best.pth'], [0, 9, 14])
            self.assertEqual(solver.state_dict.call_count, 25)
            self.assertEqual(self.read(directory, 'checkpoint.pth')['best_stat'], best['best_stat'])
            for index, event in enumerate(timeline):
                if event[0] == 'save':
                    self.assertIn(('eval', event[1]), timeline[:index])
            epoch_nine = [identity for _, state, identity in writes if state['last_epoch'] == 9]
            self.assertEqual(len(set(epoch_nine)), 1)

    def test_first_zero_ap_and_ties_keep_first_best(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, writes, _ = self.make_solver(directory, [0., 0., 0.])
            self.fit(solver)
            self.assertEqual(self.read(directory, 'best.pth')['last_epoch'], 0)
            self.assertEqual(sum(name == 'best.pth' for name, _, _ in writes), 1)

    def test_invalid_and_missing_ap_cannot_replace_valid_best(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, _, _ = self.make_solver(directory, [])
            solver.best_stat = {'epoch': 7, 'coco_eval_bbox': .6}
            for score in (float('nan'), float('inf'), -1., 1.1, 'invalid'):
                with self.subTest(score=score), redirect_stdout(io.StringIO()):
                    self.assertFalse(solver._update_best_stat({'coco_eval_bbox': [score]}, 8))
            for stats in ({}, {'coco_eval_bbox': []}, {'coco_eval_bbox': None}):
                self.assertFalse(solver._update_best_stat(stats, 8))
            self.assertEqual(solver.best_stat, {'epoch': 7, 'coco_eval_bbox': .6})

    def test_non_master_validates_but_does_not_write_or_build_state(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, writes, timeline = self.make_solver(directory, [.2] * 12, main=False)
            self.fit(solver)
            self.assertEqual(sum(event[0] == 'eval' for event in timeline), 12)
            self.assertEqual(writes, [])
            solver.state_dict.assert_not_called()
            self.assertFalse((Path(directory) / 'log.txt').exists())

    def test_best_includes_the_ema_that_was_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, _, _ = self.make_solver(directory, [.2, .4, .3], ema=True)
            self.fit(solver)
            best = self.read(directory, 'best.pth')
            self.assertEqual(best['model']['value'], 1)
            self.assertEqual(best['ema']['module']['value'], 101)
            self.assertIn('optimizer', best)
            self.assertIn('lr_scheduler', best)
            self.assertIn('scaler', best)

    def test_resume_restores_best_ap_and_accepts_only_a_new_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {'last_epoch': 9, 'model': {'value': 9},
                     'best_stat': {'epoch': 3, 'coco_eval_bbox': .7}}
            aps = [.6] * 20
            aps[13] = .8
            solver, writes, timeline = self.make_solver(directory, aps, resume=state)
            self.fit(solver)
            self.assertEqual([epoch for event, epoch in timeline if event == 'train'], list(range(10, 20)))
            self.assertEqual([checkpoint['last_epoch'] for name, checkpoint, _ in writes if name == 'best.pth'], [13])
            self.assertEqual(self.read(directory, 'best.pth')['best_stat'], {'epoch': 13, 'coco_eval_bbox': .8})

    def test_load_last_epoch_works_even_when_current_value_is_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, _, _ = self.make_solver(directory, [])
            solver.last_epoch = 0
            with redirect_stdout(io.StringIO()):
                solver.load_state_dict({'last_epoch': 9})
            self.assertEqual(solver.last_epoch, 9)

    def test_legacy_checkpoint_without_best_metadata_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            solver, _, _ = self.make_solver(directory, [.2, .3], resume={'last_epoch': 0})
            self.fit(solver)
            self.assertEqual(self.read(directory, 'best.pth')['last_epoch'], 1)

    def test_bad_checkpoint_interval_rejected(self):
        for interval in (0, -1, None, 1.5):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as directory:
                solver, _, _ = self.make_solver(directory, [.2], step=interval)
                with self.assertRaisesRegex(ValueError, 'checkpoint_step'), redirect_stdout(io.StringIO()):
                    solver.fit()


if __name__ == '__main__':
    unittest.main()
