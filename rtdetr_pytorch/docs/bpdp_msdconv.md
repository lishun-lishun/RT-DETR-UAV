# BDPD and MSDConv for RT-DETR-R18

## Audited PResNet18-d structure

For a real `1x3x640x640` forward:

| Location | Code | Output | Effective stride |
|---|---|---:|---:|
| VD stem | `conv1` | `64x320x320` | 2 |
| max pool | `F.max_pool2d` | `64x160x160` | 4 |
| S2 | `res_layers[0]`, 2 BasicBlocks | `64x160x160` | 4 |
| S3 / P3 | `res_layers[1]`, 2 BasicBlocks | `128x80x80` | 8 |
| S4 / P4 | `res_layers[2]`, 2 BasicBlocks | `256x40x40` | 16 |
| S5 / P5 | `res_layers[3]`, 2 BasicBlocks | `512x20x20` | 32 |

The real stride 4-to-8 operation is
`res_layers[1].blocks[0].branch2a`, a 64-to-128 3x3 stride-two
`ConvNormLayer`. Its VD shortcut is an independent stride-two average pool
followed by a 1x1 projection. BDPD replaces only the main-path operation; the
shortcut and residual topology remain unchanged. The next transitions are the
corresponding `res_layers[2].blocks[0].branch2a` and
`res_layers[3].blocks[0].branch2a` operations.

Pretrained PResNet weights are downloaded in `PResNet.__init__` and loaded by
state-dict key. BDPD explicitly reports the replaced
`res_layers.1.blocks.0.branch2a.*` keys. MSDConv retains every original key and
reports only its new missing keys. Any unrelated mismatch raises an error.

## BDPD-P3

The input is minimally padded on the right/bottom with replicate padding when
height or width is odd. Four phases are stacked as `[B,4,C,H/2,W/2]`. Their
mean is the base. Phase-minus-base tensors form detail. FP32 local signed and
absolute 3x3 means produce a clamped consistency ratio; detail weight is
`0.25 + 0.75 * consistency`. Base is projected `Cin->Cout`; concatenated detail
is projected `4Cin->Cout`. The bounded scalar is
`alpha_max * sigmoid(raw_alpha)`, initialized to 0.5. Fusion is followed by the
ReLU that existed in the replaced branch2a.

## MSDConv-P3

MSDConv runs after the complete S3 and before both the returned P3 and S4. G1
uses fixed separable 3x3 binomial `[1,2,1]`; G2 uses fixed 5x5
`[1,4,6,4,1]`; G3 repeats the 5x5 low-pass on G2. All kernels are registered
buffers and operate depthwise. `H=G1-G2`, `M=G2-G3`, `L=G3`. Group-wise RMS
energy routes only H/M. L enters only the two-channel context router. The
projected target band is fused as `X + beta*Z`, with
`beta=0.5*tanh(raw_beta)` initialized to 0.1.

## Verified complexity (one 640 detector forward)

PyTorch profiler values are lower bounds because not every operator has a FLOP
formula.

| Model | Params | Delta params | Profiler GFLOPs | Delta GFLOPs |
|---|---:|---:|---:|---:|
| Baseline | 20,083,028 | 0 | 61.1519 | 0 |
| BDPD-P3 | 20,050,517 | -32,511 | 60.7398 | -0.4121 |
| MSDConv-P3 | 20,099,672 | +16,644 | 61.4602 | +0.3084 |
| Combined | 20,067,161 | -15,867 | 61.0482 | -0.1037 |

## Local speed reference

Measured on an NVIDIA GeForce RTX 5060 Ti, batch 1, 640x640, AMP, 100 warmup
and 500 timed iterations. Re-run on the training server; these values are not
portable across hardware.

| Model | Mean ms | Median ms | P95 ms | FPS | Peak MiB |
|---|---:|---:|---:|---:|---:|
| Baseline | 10.353 | 10.110 | 11.403 | 96.59 | 169.70 |
| BDPD-P3 | 10.950 | 10.616 | 13.969 | 91.32 | 195.36 |
| MSDConv-P3 | 11.183 | 10.844 | 14.047 | 89.42 | 171.86 |
| Combined | 11.557 | 11.216 | 14.198 | 86.53 | 197.77 |

## Tests and offline mechanism analysis

Run synthetic, integration, backward and AMP tests:

```bash
python tools/test_bpdp_msdconv.py --test-baseline --test-bpdp --test-msdconv --test-grad --test-amp
```

Run a two-rank DDP smoke test separately for each trainable variant:

```bash
torchrun --nproc_per_node=2 tools/test_bpdp_msdconv.py --test-ddp --model bpdp
torchrun --nproc_per_node=2 tools/test_bpdp_msdconv.py --test-ddp --model msdconv
```

Use `tools/analyze_bpdp_msdconv_mechanism.py` after training. Its manifest is a
manually curated JSON list with `image`, `category`, and optional COCO-format
`boxes`. This keeps clear/blur/background labels out of training supervision.

## Stage-one commands

Use one GPU per experiment because the reported 64.6 AP Baseline was obtained
with one GPU and batch size 16. Changing only the candidate to three GPUs would
change its global batch from 16 to 48 and would not be a controlled comparison.

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp.yml --amp --seed 0
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp.yml -r output/rtdetr_r18vd_dut_anti_uav_bpdp/best.pth --test-only --amp --seed 0
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_msdconv.yml --amp --seed 0
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_msdconv.yml -r output/rtdetr_r18vd_dut_anti_uav_msdconv/best.pth --test-only --amp --seed 0
```

Both candidates inherit validation every epoch, `best.pth`, checkpoints every
10 epochs, and the combined `training_curves.png` convergence plot from the
same DUT Baseline. Do not train the combined configuration in stage one.

After a checkpoint exists, copy `docs/bpdp_msdconv_manifest.example.json`,
replace the paths/boxes with manually selected DUT samples, and run:

```bash
python tools/analyze_bpdp_msdconv_mechanism.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp.yml -r output/rtdetr_r18vd_dut_anti_uav_bpdp/best.pth --manifest docs/bpdp_msdconv_manifest.json --output output/rtdetr_r18vd_dut_anti_uav_bpdp/mechanism.json
python tools/analyze_bpdp_msdconv_mechanism.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_msdconv.yml -r output/rtdetr_r18vd_dut_anti_uav_msdconv/best.pth --manifest docs/bpdp_msdconv_manifest.json --output output/rtdetr_r18vd_dut_anti_uav_msdconv/mechanism.json
```
