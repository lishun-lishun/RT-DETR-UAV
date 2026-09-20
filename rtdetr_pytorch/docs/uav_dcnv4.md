# UAV-DCNv4 plugin

`UAVDCNv4` is the self-contained P2 plugin used by
`configs/rtdetr/rtdetr_r18vd_dut_anti_uav_p2_dcnv4.yml`. It does not import
`DCNv4.ext`, does not compile a project-specific CUDA extension, and requires
only the PyTorch/torchvision pair already used by RT-DETR.

This module is DCNv4-style rather than a bit-equivalent reimplementation of
the upstream fused DCNv4 kernel. It retains grouped learned offsets and
input-dependent spatial weights, then adds bounded offsets and explicit local
contrast evidence for small UAV targets.

## Computation

For the stride-8 S3 feature `x`, the module computes:

1. identity-initialized pointwise value projection `v`;
2. local mean `m = AvgPool3x3(v)` and high frequency `h = v - m`;
3. four groups of 3x3 offsets bounded to `[-1.5, 1.5]` feature pixels;
4. nine softmax-normalized dynamic weights per group;
5. depthwise deformable aggregation `D(v)` using
   `torchvision.ops.deform_conv2d`;
6. evidence `e = (D(v) - m) + sigmoid(g) * h`;
7. pointwise projection and normalization.

PResNet adds `alpha_eff * e`, where
`alpha_eff = 0.20 * tanh(raw_alpha)`. `raw_alpha` starts at zero, so the first
forward is exactly the original RT-DETR Baseline while gradients can still
teach the gate whether the new evidence is useful.

## YAML switch

Enabled:

```yaml
BackbonePlugins:
  P2: {enabled: true, type: uav_dcnv4, groups: 4, max_offset: 1.5, temperature: 1.0, detail_gain: 0.5, alpha_init: 0.0, alpha_max: 0.20}
```

Disabled:

```yaml
BackbonePlugins:
  P2: {enabled: false, type: uav_dcnv4}
```

Disabling the point creates no plugin module or plugin parameters. Original
PResNet pretrained weights load with only the new `plugins.*` keys missing
when the point is enabled.

## Training

No DCNv4 installation command is needed:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_p2_dcnv4.yml --amp --seed 0
```

An old checkpoint produced by the superseded native-wrapper P2 implementation
is not shape-compatible with this module. Start the UAV-DCNv4 experiment from
the same baseline/backbone initialization used by the other comparison runs.
