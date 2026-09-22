# UAV training-only tuning (RT-DETR-R18)

The original `../rtdetr_r18vd_dut_anti_uav.yml` is not overwritten. All
configs retain the original R18 backbone, HybridEncoder, transformer, matcher,
loss, EMA, 200 epochs and batch 16 **per GPU**. MERT and backbone enhancements
remain off. Run every compared experiment with the same GPU count and seed.

## Stage A: augmentation at 640

| Config | RandomZoomOut | RandomIoUCrop probability |
|---|---:|---:|
| `A0_coco_style.yml` | original 0.5 | 0.8 |
| `A1_no_zoom_crop08.yml` | off | 0.8 |
| `A2_no_zoom_crop04.yml` | off | 0.4 |
| `A3_no_zoom_crop02.yml` | off | 0.2 |
| `A4_no_zoom_no_crop.yml` | off | off |

Only output directory and the intended augmentation list differ. Do not infer
the best crop probability from dataset dimensions alone; select it by validation
AP, AP75, AP_small and AR_small. The `A2` selection below is provisional.

## Stage B: resolution after choosing augmentation

`B0_a2_640.yml`, `B1_a2_800.yml` and `B2_a2_960.yml` currently inherit A2.
If another A config wins, create corresponding B configs from that winner
*before* running them. The 800 and 960 configs update train/validation Resize,
the training multi-scale list and both evaluation spatial-size caches together.
`uav_baseline_v1.yml` is the same fixed-640 protocol as A2 but a separate
pilot output. B1/B2 are resolution-only ablations, **not** the default
external-method comparison protocol. Evaluation remains 640 for the proposed
strong baseline.
The 960 multi-scale list is capped at 1024 to limit memory demand. Check free
GPU memory before launching; do not compare experiments with different global
batch sizes without explicitly recording that difference.

## Stage C: learning rate after choosing A and B

`C0_a2_640_backbone1e5.yml` is the fixed-640 control.
`C1_a2_640_backbone2e5.yml` changes only the two backbone optimizer groups
from 1e-5 to 2e-5; head lr stays 1e-4. These configs are also provisional
until A2 wins. Queries, denoising, loss and matcher are untouched.

## Existing schedule caveat

The inherited `MultiStepLR` milestone is epoch 1000, beyond the inherited
200-epoch run, so LR does **not** decay in these first-stage experiments. The
`ema.warmups: 2000` setting is EMA warmup, **not** LR warmup. Optimize the
scheduler only after augmentation, resolution and LR comparisons. One epoch
has 325 optimizer steps on one GPU (16 images/batch) or 109 on three GPUs
(48 images/global batch, DistributedSampler padding), for 65,000 or 21,800
steps respectively over 200 epochs. Keep GPU count fixed for a fair series.

The dataset report can be reproduced with
`python tools/analyze_uav_dataset.py`; training logs with
`python tools/analyze_training_log.py output/<run>/log.txt`.
