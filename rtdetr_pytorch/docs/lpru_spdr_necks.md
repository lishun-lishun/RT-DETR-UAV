# FINAL CLEANUP & NECK REPORT

```text
================================
FINAL CLEANUP & NECK REPORT
================================
```

本报告记录 DUT-Anti-UAV 历史实验清理，以及 LPRU/SPDR 两个独立可插拔 Neck 的实现与验证状态。CPU 结果来源为 `reports/resample_necks_cpu.json`。本轮没有启动正式训练。由于本地工作区没有服务器的 `output/` 内容、CUDA 和 Bash，服务器输出清理、CUDA AMP、三卡 DDP 与 Bash dry-run 均不得视为已通过。

## 1. 删除的历史 DUT 配置文件

已从当前工程删除 13 个历史调优文件，其中 12 个 YAML 和 1 个说明文件：

```text
configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bor.yml
configs/rtdetr/rtdetr_r18vd_dut_anti_uav_dgfr.yml
configs/rtdetr/rtdetr_r18vd_dut_anti_uav_paf.yml
configs/rtdetr/rtdetr_r18vd_dut_anti_uav_slr.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_bor.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_paf.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_slr.yml
configs/rtdetr/uav_tuning/G48_lr1x.yml
configs/rtdetr/uav_tuning/G48_lr2x.yml
configs/rtdetr/uav_tuning/G48_lr3x.yml
configs/rtdetr/uav_tuning/three_gpu_base.yml
configs/rtdetr/uav_tuning/README.md
```

删除前清单保存在 `cleanup_manifest_before_new_necks.txt`。历史 Python 源码模块没有被删除。

## 2. 删除的历史 DUT 结果目录

**未在本地实际删除任何结果目录。** 当前 Windows 工作区不存在以下两个根目录，因此无法知道或安全删除 Linux 服务器上的真实历史结果：

```text
output/three_gpu_b16_warmup_cosine/
output/three_gpu_b16_warmup_cosine_test_results/
```

服务器清理仍为 **PENDING**。上传代码后先在服务器工程根目录执行只读扫描，再人工检查 manifest，最后执行清理：

```bash
python tools/cleanup_dut_outputs.py
python tools/cleanup_dut_outputs.py --apply
```

工具只允许删除两个结果根目录的直接子项，并且只有在两个受保护目录及其 `best.pth` 都存在时才允许 `--apply`。

## 3. 保留的 PResNet18 Baseline 配置/结果

已保留配置：

```text
configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml
```

其中仅加入 LPRU/SPDR 的关闭默认值；CPU disabled-equivalence 验证确认关闭时不改变原模型权重和输出。预期受保护结果为：

```text
output/three_gpu_b16_warmup_cosine/rtdetr_r18vd_dut_anti_uav/best.pth
```

该结果不在本地工作区，尚未在服务器复核其存在性；清理工具会把它作为强制保护条件。

## 4. 保留的 HRNetV2-W18 配置/结果

已保留配置：

```text
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml
```

HRNetV2-W18 的结构、ImageNet 预训练设置、P3/P4/P5 通道及 optimizer 分组均未改变。预期受保护结果为：

```text
output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav/best.pth
```

该结果同样需要在服务器执行清理 dry-run 时复核。

## 5. 新增 LPRU 源码

LPRU 实现在 `src/zoo/rtdetr/resample_neck.py` 的 `LearnablePixelReassemblyUpsample`。它保留原始 nearest 上采样 `U_base`，新增 `1x1 C→4C + PixelShuffle(2) + DWConv3x3 + 1x1` 重建路径，并使用：

```text
U = U_base + alpha * (U_learn - U_base)
alpha = alpha_max * tanh(raw_alpha)
alpha_max = 0.5, alpha_init = 0.05
```

`raw_alpha` 使用反双曲正切正确初始化；alpha 为 `[1,C,1,1]` 的逐通道 LayerScale。模块没有新增 BatchNorm、Attention、Gate、DCN、CARAFE 或 DySample。

## 6. 新增 SPDR 源码

SPDR 实现在同一文件的 `SubpixelPreservingDownsample`。它保留原 stride-2 卷积 `D_base`，新增 `PixelUnshuffle(2) + 1x1 4C→C + DWConv3x3 + 1x1 C→C` 信息保持路径，并使用：

```text
D = D_base + beta * (D_preserve - D_base)
beta = beta_max * tanh(raw_beta)
beta_max = 0.5, beta_init = 0.05
```

beta 同样是 `[1,C,1,1]` 的逐通道有界 LayerScale，且没有新增 BatchNorm。

## 7. LPRU 插入位置

LPRU 只插入 `HybridEncoder` 的 top-down 重采样路径，位于原 nearest interpolation 之后、原 concat/FPN block 之前：

```text
LPRU54: P5 → P4
LPRU43: P4 → P3
```

两处模块参数独立，不共享权重；原 lateral conv、nearest 基线路径、concat、fpn block 和 bottom-up 路径均保留。

## 8. SPDR 插入位置

SPDR 只插入 `HybridEncoder` 的 bottom-up 重采样路径，位于原 stride-2 `downsample_conv` 之后、原 concat/PAN block 之前：

```text
SPDR34: P3 → P4
SPDR45: P4 → P5
```

两处模块参数独立；top-down 仍严格使用原 nearest interpolation。配置同时启用 LPRU 与 SPDR 时会直接抛出：

```text
LPRU and SPDR must be evaluated independently in the current experiment.
```

## 9. Disabled equivalence 结果

CPU 验证结果为 **PASS**。PResNet18/HRNetV2-W18 的 LPRU 与 SPDR 四种候选配置在对应模块关闭时：

- 新 Neck 不被构造；
- `state_dict` 键和值与对应 Baseline 精确一致；
- Backbone、N3/N4/N5、`pred_logits`、`pred_boxes` 均通过 `atol=1e-6, rtol=1e-5` 的一致性检查。

说明：CPU smoke 为避免本地下载预训练权重，仅在验证进程中关闭了 backbone pretrained；正式 YAML 的预训练设置没有改变。

## 10. 480/640/800 Forward

四个新模型的 CPU detector forward 均为 **PASS**，无 NaN/Inf：

| 输入 | N3 | N4 | N5 | pred_logits | pred_boxes |
|---:|---|---|---|---|---|
| 480 | `1×256×60×60` | `1×256×30×30` | `1×256×15×15` | `1×300×1` | `1×300×4` |
| 640 | `1×256×80×80` | `1×256×40×40` | `1×256×20×20` | `1×300×1` | `1×300×4` |
| 800 | `1×256×100×100` | `1×256×50×50` | `1×256×25×25` | `1×300×1` | `1×300×4` |

## 11. Backward

四个候选模型的 CPU synthetic backward 均为 **PASS**。所有 LPRU/SPDR 新参数梯度均存在、有限且非零；测试使用动态 Neck feature，并未启动数据集训练。

## 12. AMP

状态：**NOT RUN**。本地没有 CUDA，因此不能把 CPU autocast 或静态检查写成 CUDA AMP PASS。服务器验证命令：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/validate_resample_necks.py --amp-smoke --output reports/resample_necks_amp.json
```

只有四个候选全部完成 CUDA autocast forward/backward，且无 NaN/Inf 后，才能记为 PASS。

## 13. 三卡 DDP

状态：**NOT RUN**。本地没有三张可见 CUDA GPU。服务器验证命令：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9925 tools/validate_resample_necks.py --ddp-smoke --amp --output reports/resample_necks_ddp_amp.json
```

该检查固定 `world_size=3`、`find_unused_parameters=False`，每个候选执行两步 forward/backward 和 all-reduce。只有无 unused parameter、无 hang 且三 rank 全部成功后才可记为 PASS。

## 14. Optimizer 实际 LR 和 weight_decay

CPU resolved optimizer 审计为 **PASS**。四个候选的所有新参数均进入 Neck/Encoder 主学习率，而不是 backbone 学习率：

| 参数类型 | optimizer group | LR | weight_decay |
|---|---:|---:|---:|
| `raw_alpha` / `raw_beta` | 3 | `3e-4` | `1e-4` |
| Conv/DWConv weight | 3 | `3e-4` | `1e-4` |
| Conv/DWConv bias | 2 | `3e-4` | `0` |

完整逐参数记录见 `reports/resample_necks_cpu.json` 的 `smoke.optimizer`。

## 15. Params/MACs

以下为 640×640、batch=1 的 CPU 统计。MACs 只统计实际执行的 Conv2d/Linear，是明确的下界；不包含 PixelShuffle/PixelUnshuffle、插值、归一化、激活、concat 与逐元素运算。

| 模型 | Params | ΔParams | Conv/Linear MACs 下界 | ΔMACs |
|---|---:|---:|---:|---:|
| PResNet18 Original | 20,083,028 | — | 30,006,963,200 | — |
| PResNet18 + LPRU | 20,746,580 | +663,552 | 31,073,971,200 | +1,067,008,000 |
| PResNet18 + SPDR | 20,745,044 | +662,016 | 30,666,931,200 | +659,968,000 |
| HRNetV2-W18 Original | 18,280,456 | — | 39,669,190,400 | — |
| HRNetV2-W18 + LPRU | 18,944,008 | +663,552 | 40,736,198,400 | +1,067,008,000 |
| HRNetV2-W18 + SPDR | 18,942,472 | +662,016 | 40,329,158,400 | +659,968,000 |

## 16. Fairness audit

四组 resolved-config 对比均为 **PASS**：

```text
PResNet18 Original vs PResNet18 + LPRU
PResNet18 Original vs PResNet18 + SPDR
HRNetV2-W18 Original vs HRNetV2-W18 + LPRU
HRNetV2-W18 Original vs HRNetV2-W18 + SPDR
```

允许差异只有对应 Neck namespace、`output_dir` 和必要的 `__include__` 元数据。训练参数继续继承 Baseline：200 epochs、每卡 batch 16、三卡 global batch 48、main LR `3e-4`、backbone LR `3e-5`、原 warmup/cosine/EMA、原 multi-scale、val/test 640×640、seed 0；decoder、matcher、loss、queries 和 augmentation 均未被候选 YAML 覆盖。

## 17. 最终 6 个 DUT 配置文件

当前顶层 DUT YAML 集合审计为 **PASS**，恰好为以下 6 个文件：

```text
configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml
configs/rtdetr/rtdetr_r18vd_dut_anti_uav_lpru.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_lpru.yml
configs/rtdetr/rtdetr_r18vd_dut_anti_uav_spdr.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_spdr.yml
```

公共 include 和非 DUT 官方配置不计入这 6 个正式 DUT 实验配置。

## 18. 四个新实验的 dry-run 结果

状态：**NOT RUN**。本地没有 Bash，不能把脚本静态内容写成 dry-run PASS。脚本已经固定以下顺序，未使用 sort/shuf：

```text
1. PResNet18 + LPRU
2. HRNetV2-W18 + LPRU
3. PResNet18 + SPDR
4. HRNetV2-W18 + SPDR
```

服务器执行：

```bash
bash tools/train_all_dut_modules_3gpu.sh --dry-run
```

必须确认输出含 `Total experiments = 4`、上述顺序和正确的 RUN/SKIP/output_dir，且没有创建目录，才能记为 PASS。

## 19. 三卡连续训练状态

状态：**NOT STARTED**。本轮严格未启动训练。只有服务器历史输出清理、AMP、DDP 和 dry-run 都通过后，才执行：

```bash
bash tools/train_all_dut_modules_3gpu.sh
```

脚本使用 `CUDA_VISIBLE_DEVICES=1,2,3`、`--nproc_per_node=3`、`--amp --seed 0`，按固定四实验顺序串行运行；目标 output 已存在时会 SKIP。单个实验失败会被记录并继续下一个实验，四项处理完毕后，只要存在失败，脚本就以非零状态返回。

## 20. 统一 Test 结果

状态：**N/A — 尚未训练四个新实验，因此没有可报告的新 `best.pth` 或指标。** 不得用 CPU smoke、validation 或旧历史结果冒充正式 Test。

训练完成后使用：

```bash
GPU_ID=1 NUM_WORKERS=2 bash tools/test_all_best_dut.sh
```

统一协议为 `split=test`、640×640、EMA、FP32 evaluate、`num_workers=2`，输出 AP、AP50、AP75、APS/APM/APL、AR1/AR10/AR100、ARS/ARM/ARL。最终结果表在正式测试完成前保持空白：

| Model | AP | AP50 | AP75 | APS | APM | APL | AR100 | ARS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PResNet18 | pending server result audit | | | | | | | |
| PResNet18 + LPRU | not trained | | | | | | | |
| PResNet18 + SPDR | not trained | | | | | | | |
| HRNetV2-W18 | pending server result audit | | | | | | | |
| HRNetV2-W18 + LPRU | not trained | | | | | | | |
| HRNetV2-W18 + SPDR | not trained | | | | | | | |

## 最终状态标志

```text
Historical DUT configs cleaned: YES

Historical DUT outputs cleaned: NO (server pending)

Baseline preserved: YES (config/source; server checkpoint pending audit)

HRNet preserved: YES (config/source; server checkpoint pending audit)

LPRU pluggable: YES

SPDR pluggable: YES

LPRU disabled restores original behavior: YES

SPDR disabled restores original behavior: YES

Training hyperparameters unchanged: YES

Four experiments ready: NO (CPU/config ready; server cleanup, AMP, DDP and Bash dry-run pending)
```
