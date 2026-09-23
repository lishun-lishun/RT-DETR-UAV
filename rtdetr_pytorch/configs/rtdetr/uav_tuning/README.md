# DUT-Anti-UAV three-GPU learning-rate search

These configs are for exactly three DDP processes. `batch_size: 16` is kept
unchanged **per GPU**, therefore the global batch is 48. The original
augmentation, 640 validation protocol, loss, model, 200 epochs and checkpoint
policy remain unchanged. Only learning rate is varied; all runs use the same
EMA ramp adjusted from 2000 to 667 optimizer updates.

| Config | Main LR | Backbone LR | Purpose |
|---|---:|---:|---|
| `G48_lr1x.yml` | 1e-4 | 1e-5 | unchanged-LR control |
| `G48_lr2x.yml` | 2e-4 | 2e-5 | conservative candidate; run first |
| `G48_lr3x.yml` | 3e-4 | 3e-5 | linear batch-size scaling |

Run all three with the same seed and GPUs. Select by validation mAP50-95,
using mAP75 as the localization tie-breaker. Do not select by AP50 alone.
The inherited scheduler milestone is still beyond 200 epochs so this first
search changes only LR, not its schedule. Tune decay only after choosing LR.

Every epoch updates `training_curves.png` in that run's output directory. The
single figure contains train loss, validation mAP50, mAP75 and mAP50-95.

Example:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9909 tools/train.py -c configs/rtdetr/uav_tuning/G48_lr2x.yml --amp --seed 0
```
