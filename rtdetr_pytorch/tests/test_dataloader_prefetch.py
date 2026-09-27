"""Regression tests for bounded shared-memory prefetch under DDP."""

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import SequentialSampler, TensorDataset

from tests._support import PROJECT_DIR, prepare_imports
prepare_imports()

from src.data.dataloader import DataLoader  # noqa: E402
import src.misc.dist as dist  # noqa: E402
from tools.analyze_dut_models import fresh_config  # noqa: E402


class DataLoaderPrefetchTests(unittest.TestCase):
    def test_dut_protocol_keeps_workers_and_uses_one_batch_prefetch(self):
        config = fresh_config(
            PROJECT_DIR / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml')
        self.assertEqual(config['train_dataloader']['num_workers'], 4)
        self.assertEqual(config['train_dataloader']['prefetch_factor'], 1)
        self.assertFalse(config['train_dataloader']['persistent_workers'])
        self.assertEqual(config['val_dataloader']['num_workers'], 2)
        self.assertEqual(config['val_dataloader']['prefetch_factor'], 1)

    def test_ddp_rebuild_preserves_prefetch_and_persistence(self):
        dataset = TensorDataset(torch.arange(8))
        loader = DataLoader(
            dataset, batch_size=2, num_workers=2, prefetch_factor=1,
            persistent_workers=False)
        with patch.object(dist, 'is_dist_available_and_initialized',
                          return_value=True), \
                patch.object(dist, 'DistributedSampler',
                             side_effect=lambda data, shuffle=False:
                             SequentialSampler(data)):
            rebuilt = dist.warp_loader(loader, shuffle=True)
        self.assertEqual(rebuilt.num_workers, 2)
        self.assertEqual(rebuilt.prefetch_factor, 1)
        self.assertFalse(rebuilt.persistent_workers)


if __name__ == '__main__':
    unittest.main()
