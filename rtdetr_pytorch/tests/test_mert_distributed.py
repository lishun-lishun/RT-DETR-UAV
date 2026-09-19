"""Two-step, two-rank CPU/Gloo smoke test of real concatenated MERT DDP.

Only synthetic 128px images are used. Production configs, SyncBN, datasets,
loggers and AMP behavior are untouched; native criterion normalization runs.
"""

from datetime import timedelta
from pathlib import Path
import tempfile
import time
import unittest

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing


PROJECT = Path(__file__).resolve().parents[1]


def _run_rank(rank, rendezvous):
    from tests._support import prepare_imports
    prepare_imports()
    from src.core import YAMLConfig
    from src.solver.det_engine import _forward_mert_views, _mert_train_losses
    from src.solver.mert import MERT

    torch.set_num_threads(1)
    torch.manual_seed(91 + rank)
    distributed.init_process_group('gloo', rank=rank, world_size=2,
                                   init_method=Path(rendezvous).resolve().as_uri(),
                                   timeout=timedelta(seconds=40))
    try:
        cfg = YAMLConfig(str(PROJECT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_34_mert_late_xywh.yml'))
        cfg.yaml_cfg['PResNet']['pretrained'] = False  # Test only; no downloads.
        detector = cfg.model
        detector.multi_scale = None  # Test only; enough anchors for 300 queries.
        model = torch.nn.parallel.DistributedDataParallel(detector,
                                                          find_unused_parameters=True)
        model.train()
        criterion = cfg.criterion.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        mert = MERT(cfg.yaml_cfg['MERT'])
        assert mert.forward_mode == 'concat'
        calls = {'ddp': 0, 'detector': 0, 'secd': 0}
        batches = []

        def count_ddp(module, inputs, outputs):
            calls['ddp'] += 1

        def count_detector(module, inputs, outputs):
            calls['detector'] += 1
            assert inputs[0].shape[0] == 2

        def count_secd(module, inputs, outputs):
            calls['secd'] += 1
            batches.append(inputs[0].shape[0])

        hooks = [model.register_forward_hook(count_ddp),
                 detector.register_forward_hook(count_detector),
                 detector.backbone.secd_34.register_forward_hook(count_secd)]
        for step in range(2):
            samples = torch.randn(1, 3, 128, 128)
            boxes = [[0.5, 0.5, 0.1, 0.1]]
            if rank == 1:
                # Initially inside, then partially clipped by the +1px shift:
                # still detection supervision, excluded from trajectory loss.
                boxes.append([0.992, 0.5, 0.015, 0.1])
            targets = [{'boxes': torch.tensor(boxes),
                        'labels': torch.zeros(len(boxes), dtype=torch.long),
                        'image_id': torch.tensor([rank]),
                        'orig_size': torch.tensor([128, 128]),
                        'size': torch.tensor([128, 128])}]
            pair = mert.prepare(samples, targets, model, shifts=[[1, 0]])
            assert len(pair.targets[0]['boxes']) == len(pair.shifted_targets[0]['boxes'])
            if rank == 1:
                assert pair.targets[0]['fully_visible'].all()
                assert not pair.shifted_targets[0]['fully_visible'][-1]
            outputs = _forward_mert_views(model, pair, mert)
            assert outputs['dn_meta']['dn_num_group'] > 0
            assert outputs['dn_aux_outputs']
            losses = _mert_train_losses(outputs, pair, mert, criterion)
            assert any('_dn_' in name or name.endswith('_dn') for name in losses)
            assert losses['loss_mert'].ndim == 0 and losses['loss_mert'].requires_grad
            loss = sum(losses.values())
            assert torch.isfinite(loss), (rank, step, losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, (rank, step, 'missing gradient', name)
                    assert torch.isfinite(parameter.grad).all(), (rank, step, 'nonfinite gradient', name)
            optimizer.step()
            assert calls == {'ddp': step + 1, 'detector': step + 1, 'secd': step + 1}, calls
        assert batches == [2, 2]

        # Regression for global pair normalization: rank 0 contributes one
        # low-error pair while rank 1 contributes two high-error pairs. The
        # averaged DDP loss must equal the global three-pair mean, not the mean
        # of two independently normalized rank means.
        audit_mert = MERT({'enabled': True,
                           'small_object_weighting': {'enabled': False}})
        pair_count = rank + 1
        audit_targets = [{'boxes': torch.tensor([[.3 + .2 * i, .5, .1, .1]
                                                 for i in range(pair_count)]),
                          'labels': torch.zeros(pair_count, dtype=torch.long),
                          'image_id': torch.tensor([rank]),
                          'orig_size': torch.tensor([128, 128]),
                          'size': torch.tensor([128, 128])}]
        audit_pair = audit_mert.prepare(torch.zeros(1, 3, 128, 128), audit_targets,
                                        detector, shifts=[[1, 0]])
        base = audit_pair.targets[0]['boxes']
        original_trajectory = base[None, None].expand(3, 1, pair_count, 4).clone()
        shifted_trajectory = original_trajectory.clone()
        shifted_trajectory[..., 0] += 1 / 128
        error = .02 if rank == 0 else .04
        shifted_trajectory[1:, ..., 0] += torch.tensor([error, 2 * error])[:, None, None]

        def outputs(trajectory):
            logits = torch.zeros(1, pair_count, 1)
            return {'pred_boxes': trajectory[-1], 'pred_logits': logits,
                    'aux_outputs': [
                        {'pred_boxes': trajectory[0], 'pred_logits': logits},
                        {'pred_boxes': trajectory[1], 'pred_logits': logits},
                        {'pred_boxes': torch.zeros_like(trajectory[0]), 'pred_logits': logits}]}

        class IdentityMatcher:
            def __call__(self, output, targets):
                indices = torch.arange(len(targets[0]['boxes']))
                return [(indices, indices)]

        audit_loss = audit_mert.calculate_loss(outputs(original_trajectory),
                                               outputs(shifted_trajectory),
                                               audit_pair, IdentityMatcher())['loss_mert']
        averaged_loss = audit_loss.detach().clone()
        distributed.all_reduce(averaged_loss)
        averaged_loss /= distributed.get_world_size()
        expected = .1 * (torch.nn.functional.smooth_l1_loss(
            torch.tensor(0.), torch.tensor(.02), beta=.01) +
            2 * torch.nn.functional.smooth_l1_loss(
                torch.tensor(0.), torch.tensor(.04), beta=.01)) / 3
        assert torch.allclose(averaged_loss, expected), (rank, audit_loss, averaged_loss, expected)
        for hook in hooks:
            hook.remove()
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


@unittest.skipUnless(distributed.is_available() and distributed.is_gloo_available(),
                     'CPU/Gloo distributed backend is unavailable')
class MERTDistributedTests(unittest.TestCase):
    def test_real_r18_secd_mert_concat_ddp_two_steps_two_ranks(self):
        with tempfile.TemporaryDirectory(prefix='mert-ddp-test-') as folder:
            rendezvous = str(Path(folder) / 'rendezvous')
            context = multiprocessing.spawn(_run_rank, args=(rendezvous,),
                                            nprocs=2, join=False)
            deadline = time.monotonic() + 45
            try:
                while not context.join(timeout=max(0, deadline - time.monotonic())):
                    if time.monotonic() >= deadline:
                        self.fail('Two-rank MERT smoke test exceeded its 45-second deadline')
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
