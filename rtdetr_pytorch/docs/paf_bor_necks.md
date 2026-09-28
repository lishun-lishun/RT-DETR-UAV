# PAF/BOR Neck implementation report

==============================
FINAL IMPLEMENTATION REPORT
==============================

## 1. Added and modified files

Added:

- `src/zoo/rtdetr/paf_neck.py`
- `src/zoo/rtdetr/bor_neck.py`
- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_paf.yml`
- `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_paf.yml`
- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bor.yml`
- `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_bor.yml`
- `tests/test_paf_neck_unit.py`
- `tests/test_bor_neck.py`
- `tests/test_paf_bor_integration.py`
- `tools/validate_paf_bor_necks.py`
- `reports/paf_bor_validation_cpu.json`

Modified:

- `src/zoo/rtdetr/hybrid_encoder.py`
- `src/core/yaml_config.py`
- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml` (default-off switches only)
- `tools/train_all_dut_modules_3gpu.sh`
- `tests/test_slr_neck.py` (expected formal YAML set only)

No backbone, decoder, criterion, matcher, dataset transform, loss, input-size,
optimizer or scheduler implementation was changed.

## 2. PAF code location

`PhaseAdaptiveFusion` is implemented in `src/zoo/rtdetr/paf_neck.py`.
It creates phases 00/01/10/11 with zero-padding plus crop (never
`torch.roll`), forms Q from the shallow feature and K from each phase, applies
FP32 four-phase softmax, and returns the convex combination of the original
upsampled features. There is no value projection, gate, residual scale, Norm,
extra loss or amplitude gate.

## 3. BOR code location

`BackgroundOrthogonalResidual` is implemented in
`src/zoo/rtdetr/bor_neck.py`. It computes the 7x7-minus-3x3 ring prototype,
vectorized background-parallel/orthogonal components, novelty ratio, fixed
theta/tau gate, one 1x1 projection and bounded residual alpha. Statistics,
division and norms stay FP32 under AMP; theta and tau are not learnable.

## 4. Exact PAF insertion point

The original HybridEncoder still runs `input_proj`, AIFI and lateral
projection. Immediately after each original nearest-neighbour `Upsample` and
before `concat(shallow, upsampled)` / the unchanged `CSPRepLayer`, PAF may
replace the upsampled tensor with its phase-aligned tensor. Independent
`paf54` and `paf43` modules serve P5->P4 and P4->P3. Bottom-up PAN is unchanged.

## 5. Exact BOR insertion point

The complete original top-down and bottom-up CCFF first produces N3/N4/N5.
BOR is then applied only as `N3 = BOR(N3)`. N4 and N5 bypass BOR exactly.

## 6. Disabled baseline equivalence

- PAF disabled versus original HybridEncoder: **PASS**
- BOR disabled versus original HybridEncoder: **PASS**
- PResNet18 backbone outputs, N3/N4/N5, `pred_logits`, `pred_boxes`: **PASS**
- HRNetV2-W18 backbone outputs, N3/N4/N5, `pred_logits`, `pred_boxes`: **PASS**
- Tolerance: `atol=1e-6`, `rtol=1e-5`
- Disabled mode constructs no PAF/BOR parameter or module.

## 7. PResNet18 + PAF

- 480/640/800 backbone + encoder forward: **PASS**
- 640 detector forward: **PASS**
- Backward and all PAF gradients finite/non-zero: **PASS**
- CUDA AMP: **NOT RUN locally (CPU-only environment); server command below**
- Three-GPU DDP: **NOT RUN locally; server command below**

## 8. HRNetV2-W18 + PAF

- 480/640/800 backbone + encoder forward: **PASS**
- 640 detector forward: **PASS**
- Backward and all PAF gradients finite/non-zero: **PASS**
- CUDA AMP: **NOT RUN locally**
- Three-GPU DDP: **NOT RUN locally**

## 9. PResNet18 + BOR

- 480/640/800 backbone + encoder forward: **PASS**
- 640 detector forward: **PASS**
- Backward and all BOR gradients finite/non-zero: **PASS**
- Effective alpha at initialization: approximately **0.05**
- CUDA AMP: **NOT RUN locally**
- Three-GPU DDP: **NOT RUN locally**

## 10. HRNetV2-W18 + BOR

- 480/640/800 backbone + encoder forward: **PASS**
- 640 detector forward: **PASS**
- Backward and all BOR gradients finite/non-zero: **PASS**
- Effective alpha at initialization: approximately **0.05**
- CUDA AMP: **NOT RUN locally**
- Three-GPU DDP: **NOT RUN locally**

## 11. Optimizer groups

Actual optimizer construction was inspected for all four models:

- PAF Wq/Wk weights: LR `3e-4`, weight decay `1e-4`
- PAF Wq/Wk biases: LR `3e-4`, weight decay `0`
- BOR projection weight: LR `3e-4`, weight decay `1e-4`
- BOR projection bias: LR `3e-4`, weight decay `0`
- BOR `raw_alpha`: LR `3e-4`, weight decay `1e-4`

No new parameter enters the backbone LR (`3e-5`) group. The current HRNet
optimizer/BatchNorm treatment was not changed.

## 12. Params and 640x640 MAC lower bounds

MACs below count executed Conv2d/Linear operations. Functional interpolation,
pooling, phase attention, softmax, vector norms, division, gates and
elementwise operations are explicitly not counted, so these are reproducible
lower bounds rather than total theoretical MACs.

| Model | Whole params | Conv/Linear MAC lower bound | Delta params | Delta MACs |
|---|---:|---:|---:|---:|
| PResNet18 Original | 20,083,028 | 30,006,963,200 | - | - |
| PResNet18 + PAF | 20,115,924 | 30,334,643,200 | +32,896 | +327,680,000 |
| PResNet18 + BOR | 20,148,821 | 30,426,393,600 | +65,793 | +419,430,400 |
| HRNetV2-W18 Original | 18,280,456 | 39,669,190,400 | - | - |
| HRNetV2-W18 + PAF | 18,313,352 | 39,996,870,400 | +32,896 | +327,680,000 |
| HRNetV2-W18 + BOR | 18,346,249 | 40,088,620,800 | +65,793 | +419,430,400 |

## 13. Resolved-config fairness audit

All four comparisons passed. The only resolved differences are the active
PAF or BOR namespace, `output_dir`, and include metadata. Epochs, per-rank and
global batch, optimizer, main/backbone LR, weight decay, warmup cosine
scheduler, EMA, AMP command, seed command, 480-800 multi-scale, 640
validation/test size, augmentation, DataLoader, decoder, denoising, matcher,
loss, queries and HybridEncoder hidden dimension are unchanged.

## 14. Four new YAML paths

1. `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_paf.yml`
2. `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_paf.yml`
3. `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bor.yml`
4. `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_bor.yml`

## 15. Batch order

Following the later request to run six experiments together, the queue is now:

1. PResNet18 + PAF
2. HRNetV2-W18 + PAF
3. PResNet18 + BOR
4. HRNetV2-W18 + BOR
5. PResNet18 + SLR
6. HRNetV2-W18 + SLR

Existing final output directories are skipped. Baseline, HRNet Original and
ACR are not added to this queue.

## 16. Dry-run result

`bash tools/train_all_dut_modules_3gpu.sh --dry-run` passed with:

- `Total configs: 6`
- fixed order matching the list above
- six RUN entries in the local empty-output checkout
- no training process or output directory created

## 17. Commands

CPU validation already executed:

```bash
python tools/validate_paf_bor_necks.py --smoke --complexity --output reports/paf_bor_validation_cpu.json
```

Server CUDA AMP validation:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/validate_paf_bor_necks.py --amp-smoke
```

Server three-GPU DDP/AMP smoke validation:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9923 tools/validate_paf_bor_necks.py --ddp-smoke --amp
```

Six-experiment sequential training queue:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 bash tools/train_all_dut_modules_3gpu.sh
```

PAF pluggable: **YES**

BOR pluggable: **YES**

PAF disabled restores original model: **YES**

BOR disabled restores original model: **YES**

Training parameters changed: **NO**

Four new experiments ready: **YES**

Formal long training started: **NO**
