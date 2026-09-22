"""Distributed BAFR/HCBR backward smoke test (no dataset or training).

From rtdetr_pytorch on a CUDA server:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=9917 tools/ddp_smoke_bafr_hcbr.py

Both ranks execute all three backbone variants with a small synthetic input.
This checks that DDP finds all trainable parameters and gradients are finite;
it does not substitute for a multi-GPU training run.
"""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nccl', 'gloo'),
                        default='nccl' if os.name != 'nt' else 'gloo')
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from tools.analyze_dut_models import import_model_source

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.backend == 'nccl':
        if not torch.cuda.is_available():
            raise RuntimeError('NCCL requires CUDA; use --backend gloo for CPU smoke')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    torch.set_num_threads(2)
    dist.init_process_group(args.backend, init_method='env://')
    try:
        core = import_model_source(selective=True)
        for suffix in ('bafr', 'hcbr', 'bafr_hcbr'):
            path = ROOT / 'configs' / 'rtdetr' / (
                f'rtdetr_r18vd_dut_anti_uav_{suffix}.yml')
            torch.manual_seed(31 + rank)
            model = core.YAMLConfig(str(path), PResNet={'pretrained': False}).model
            backbone = model.backbone.to(device).train()
            wrapped = DistributedDataParallel(
                backbone, device_ids=[local_rank] if device.type == 'cuda' else None,
                find_unused_parameters=False)
            sample = torch.randn(1, 3, 128, 128, device=device)
            levels = wrapped(sample)
            loss = sum(level.float().square().mean() for level in levels)
            loss.backward()
            missing = [name for name, parameter in wrapped.module.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            nonfinite = [name for name, parameter in wrapped.module.named_parameters()
                         if parameter.grad is not None
                         and not bool(torch.isfinite(parameter.grad).all())]
            if missing or nonfinite:
                raise RuntimeError(f'{suffix} rank={rank}: missing_grad={missing}, '
                                   f'nonfinite_grad={nonfinite}')
            dist.barrier()
            if rank == 0:
                print(f'{suffix}: DDP forward/backward OK on '
                      f'{dist.get_world_size()} rank(s); loss={loss.item():.6f}', flush=True)
            del wrapped, backbone, model, levels, sample, loss
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
