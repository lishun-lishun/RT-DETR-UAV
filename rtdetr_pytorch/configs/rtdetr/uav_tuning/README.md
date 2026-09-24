# DUT-Anti-UAV three-GPU learning-rate records

These optional ablations are for exactly three DDP processes. `batch_size: 16`
is kept unchanged **per GPU**, therefore the global batch is 48. The selected
default protocol now lives in `rtdetr_r18vd_dut_anti_uav.yml`: 3x peak LR,
five warmup epochs, proportional cosine decay, and EMA warmup 667. These files
remain available only for explicit 1x/2x/3x LR comparisons.

| Config | Main LR | Backbone LR | Purpose |
|---|---:|---:|---|
| `G48_lr1x.yml` | 1e-4 | 1e-5 | unchanged-LR control |
| `G48_lr2x.yml` | 2e-4 | 2e-5 | conservative ablation |
| `G48_lr3x.yml` | 3e-4 | 3e-5 | selected linear batch scaling |

If rerunning the ablation, use the same seed and GPUs. Select by validation
mAP50-95, using mAP75 as the localization tie-breaker. Do not select by AP50
alone. All three now inherit the same warmup-cosine schedule shape.

Every epoch updates `training_curves.png` in that run's output directory. The
single large figure contains six subplots: loss, LR, mAP50-95, mAP50, mAP75,
and AP-small.

Example:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9909 tools/train.py -c configs/rtdetr/uav_tuning/G48_lr2x.yml --amp --seed 0
```
