"""Two-rank Gloo regression for MERT's global valid-pair normalization.

This test imports only the real ``src/solver/mert.py`` file. It deliberately
needs no dataset, torchvision, SciPy matcher, YAML, detector, or GPU, so the
normalization/deadlock property can be audited in a minimal environment.
"""

from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing


PROJECT = Path(__file__).resolve().parents[1]


def load_mert_module():
    spec = importlib.util.spec_from_file_location('_mert_ddp_real',
                                                  PROJECT / 'src/solver/mert.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def outputs(trajectory):
    batch, queries = trajectory.shape[1:3]
    logits = torch.zeros(batch, queries, 1)
    return {'pred_boxes': trajectory[-1], 'pred_logits': logits,
            'aux_outputs': [
                {'pred_boxes': trajectory[0], 'pred_logits': logits},
                {'pred_boxes': trajectory[1], 'pred_logits': logits},
                # Original decoder protocol: last aux item is encoder output.
                {'pred_boxes': torch.zeros_like(trajectory[0]), 'pred_logits': logits}]}


class IdentityMatcher:
    def __call__(self, output, targets):
        result = []
        for target in targets:
            index = torch.arange(len(target['boxes']))
            result.append((index, index))
        return result


def batch(module, count, requires_grad=True):
    boxes = torch.tensor([[.25 + .2 * index, .5, .1, .1]
                          for index in range(count)]).reshape(count, 4)
    target = {'boxes': boxes, 'labels': torch.zeros(count, dtype=torch.long),
              'origin_gt_id': torch.arange(count),
              'fully_visible': torch.ones(count, dtype=torch.bool)}
    trajectory = boxes[None, None].expand(3, 1, count, 4).clone()
    if requires_grad:
        trajectory.requires_grad_()
    pair = module.MERTBatch(torch.zeros(1, 3, 128, 128), [target],
                            torch.zeros(1, 3, 128, 128), [dict(target)],
                            torch.tensor([[1, 0]]), (128, 128))
    return pair, trajectory


def run_rank(rank, rendezvous):
    module = load_mert_module()
    distributed.init_process_group('gloo', rank=rank, world_size=2,
                                   init_method=rendezvous,
                                   timeout=timedelta(seconds=20))
    try:
        mert = module.MERT({'enabled': True,
                            'small_object_weighting': {'enabled': False}})
        count = rank + 1
        pair, original = batch(module, count)
        shifted = original.detach().clone()
        shifted[..., 0] += 1 / 128
        error = .02 if rank == 0 else .04
        shifted[1:, ..., 0] += torch.tensor([error, 2 * error])[:, None, None]
        shifted.requires_grad_()
        loss = mert.calculate_loss(outputs(original), outputs(shifted),
                                   pair, IdentityMatcher())['loss_mert']
        loss.backward()
        assert original.grad is not None and shifted.grad is not None
        averaged = loss.detach().clone()
        distributed.all_reduce(averaged)
        averaged /= 2
        expected = .1 * (torch.nn.functional.smooth_l1_loss(
            torch.tensor(0.), torch.tensor(.02), beta=.01) +
            2 * torch.nn.functional.smooth_l1_loss(
                torch.tensor(0.), torch.tensor(.04), beta=.01)) / 3
        assert torch.allclose(averaged, expected), (rank, averaged, expected)

        # Rank 0 has no pair while rank 1 has one. Both ranks must reach the
        # internal pair-count all_reduce, and the empty rank keeps a connected
        # differentiable zero instead of hanging its peer.
        count = rank
        pair, original = batch(module, count)
        shifted = original.detach().clone().requires_grad_()
        loss = mert.calculate_loss(outputs(original), outputs(shifted),
                                   pair, IdentityMatcher())['loss_mert']
        loss.backward()
        assert original.grad is not None and shifted.grad is not None
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


@unittest.skipUnless(os.name != 'nt' and distributed.is_available()
                     and distributed.is_gloo_available(),
                     'This minimal two-rank Gloo regression runs on the Linux training server')
class MERTDDPReductionTest(unittest.TestCase):
    def test_two_rank_global_pair_mean_and_empty_rank(self):
        with tempfile.TemporaryDirectory(prefix='mert-ddp-reduction-') as folder:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                rendezvous = f'tcp://127.0.0.1:{listener.getsockname()[1]}'
            context = multiprocessing.spawn(run_rank, args=(rendezvous,),
                                            nprocs=2, join=False)
            deadline = time.monotonic() + 30
            try:
                while not context.join(timeout=max(0, deadline - time.monotonic())):
                    if time.monotonic() >= deadline:
                        self.fail('MERT DDP reduction test exceeded 30 seconds')
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
