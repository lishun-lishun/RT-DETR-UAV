# ==================================================
# FINAL IMPLEMENTATION REPORT
# ==================================================

## 1. 保留的 4 个 DUT 正式实验配置

- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml`
- `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml`
- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_slr.yml`
- `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_slr.yml`

Baseline 与 HRNet 原配置内容未修改。两个 SLR 配置分别继承对应对照组，只覆盖
`SLR` 和 `output_dir`。

## 2. 删除的历史 DUT 实验配置

仅删除下列顶层实验 YAML，没有删除源码、权重、output、checkpoint、日志或数据：

- `rtdetr_hrnetv2_w18_dut_anti_uav_acr.yml`
- `rtdetr_r18vd_dut_anti_uav_acr.yml`
- `rtdetr_r18vd_dut_anti_uav_akconv.yml`
- `rtdetr_r18vd_dut_anti_uav_bafr.yml`
- `rtdetr_r18vd_dut_anti_uav_bafr_hcbr.yml`
- `rtdetr_r18vd_dut_anti_uav_bpdp.yml`
- `rtdetr_r18vd_dut_anti_uav_bpdp_msdconv.yml`
- `rtdetr_r18vd_dut_anti_uav_cced34.yml`
- `rtdetr_r18vd_dut_anti_uav_cced34_grer34.yml`
- `rtdetr_r18vd_dut_anti_uav_deconv.yml`
- `rtdetr_r18vd_dut_anti_uav_drb.yml`
- `rtdetr_r18vd_dut_anti_uav_grer34.yml`
- `rtdetr_r18vd_dut_anti_uav_hcbr.yml`
- `rtdetr_r18vd_dut_anti_uav_hsdr_a.yml`
- `rtdetr_r18vd_dut_anti_uav_hsdr_b.yml`
- `rtdetr_r18vd_dut_anti_uav_mert_late_xywh.yml`
- `rtdetr_r18vd_dut_anti_uav_msdconv.yml`
- `rtdetr_r18vd_dut_anti_uav_p0_srfd.yml`
- `rtdetr_r18vd_dut_anti_uav_p1_deconv.yml`
- `rtdetr_r18vd_dut_anti_uav_p2_dcnv4.yml`
- `rtdetr_r18vd_dut_anti_uav_p3_secd34.yml`
- `rtdetr_r18vd_dut_anti_uav_p4_fadc.yml`
- `rtdetr_r18vd_dut_anti_uav_p4_wtconv.yml`
- `rtdetr_r18vd_dut_anti_uav_pdr3.yml`
- `rtdetr_r18vd_dut_anti_uav_pdr34.yml`
- `rtdetr_r18vd_dut_anti_uav_pdr34_nogate.yml`
- `rtdetr_r18vd_dut_anti_uav_phsb.yml`
- `rtdetr_r18vd_dut_anti_uav_rfaconv.yml`
- `rtdetr_r18vd_dut_anti_uav_secd_34.yml`
- `rtdetr_r18vd_dut_anti_uav_secd_34_mert_late_xywh.yml`
- `rtdetr_r18vd_dut_anti_uav_secd_345.yml`
- `rtdetr_r18vd_dut_anti_uav_secd_345_mert_late_xywh.yml`
- `rtdetr_r18vd_dut_anti_uav_secd_45.yml`
- `rtdetr_r18vd_dut_anti_uav_srfd.yml`

公共依赖 `rtdetr_r18vd_6x_coco.yml`、dataset、runtime、dataloader、optimizer、
`rtdetr_r50vd.yml`、`hrnetv2_w18.yml` 均保留。

## 3. 保留的历史源码

MERT、SECD、PDR、BAFR、BDPD、MSDConv、GRER、CCED、插件点及 ACR 等 Python
源码仍然存在。此次清理只针对 DUT 实验 YAML，Git 可恢复全部被删配置。

## 4. SLR 代码位置与接口

- 文件：`src/zoo/rtdetr/slr_neck.py`
- 类：`SLRNeck`
- 调用：`HybridEncoder.forward()` 完成原始 input projection、AIFI、Top-down FPN
  和 Bottom-up PAN 后，执行 `N3' = SLRNeck(N3, detail)`。
- Backbone 在关闭时继续返回原 list；开启时返回
  `{'features': [P3,P4,P5], 'detail': detail_source}`。没有 hook 或全局变量。

SLR 使用无损 `pixel_unshuffle(detail, 2)`，四相顺序为 TL/TR/BL/BR；Query
来自原 N3，Key 来自四个 detail 子单元，Value 只来自减去四相均值后的局部残差。
Softmax 仅沿四相维度，位置坐标明确为 `(x,y)`。

## 5. Detail Source

- PResNet18：完整 S2/C2 输出，64 channels，stride 4。
- HRNetV2-W18：Stage4 最终 `/4` 高分辨率分支，18 channels，stride 4。

原检测输出保持：PResNet `128/256/512 @ /8,/16,/32`；HRNet
`36/72/144 @ /8,/16,/32`。

## 6. SLR 输出

- `N3 -> N3 + alpha * Rloc`
- `N4` unchanged
- `N5` unchanged
- `alpha_eff = 0.0500000045`（目标初值 0.05，`alpha_max=0.3`）
- Decoder 仍只接收三个层级，没有 P2/P6。

## 7. Shape 测试

| 输入 | PResNet detail | HRNet detail | Encoder 输出 | 结果 |
|---|---|---|---|---|
| 480 | `[1,64,120,120]` | `[1,18,120,120]` | `60²,30²,15²` | PASS |
| 640 | `[1,64,160,160]` | `[1,18,160,160]` | `80²,40²,20²` | PASS |
| 800 | `[1,64,200,200]` | `[1,18,200,200]` | `100²,50²,25²` | PASS |

640 完整 Detector 输出仍为 logits `[1,300,1]`、boxes `[1,300,4]`。

## 8. Baseline equivalence

`SLR.enabled=false`：原 HybridEncoder N3/N4/N5 与最终预测均通过
`atol=1e-6, rtol=1e-5`；state dict 键和值相同。**PASS**。

同 seed 构建 SLR candidate 后，所有与对照组同名的参数/缓冲区仍逐 tensor 完全相等；
SLR 初始化使用隔离 RNG，不会改变原 Encoder/Decoder 初始化。

## 9. Optimizer

所有 `encoder.slr.*` 参数实际 LR 均为 `3e-4`：

- weights、`raw_alpha`：weight decay `1e-4`
- Conv biases：weight decay `0`
- 没有参数进入 backbone `3e-5` 组
- SLR 内未增加 Norm

## 10. Config 公平性

- Baseline vs PResNet18+SLR：合法差异仅 `SLR.*`、`output_dir`、include 元数据，YES。
- HRNet vs HRNet+SLR：合法差异仅 `SLR.*`、`output_dir`、include 元数据，YES。

Epoch 200、每卡 batch 16、global batch 48、三卡、主 LR `3e-4`、backbone LR
`3e-5`、optimizer、scheduler、EMA、480–800 multi-scale、Val/Test 640、数据增强、
Decoder、Matcher、Loss、queries、denoising 均未修改。

## 11. Params/MACs（640）

| 模型 | Whole Params | Whole Conv/Linear MACs | 新增 Params | 新增 Conv/Linear MACs |
|---|---:|---:|---:|---:|
| PResNet18 Original | 20,083,028 | 30,006,963,200 | – | – |
| PResNet18 + SLR | 20,108,005 | 30,243,097,600 | 24,977 | 236,134,400 |
| HRNet Original | 18,280,456 | 39,669,190,400 | – | – |
| HRNet + SLR | 18,302,489 | 39,829,958,400 | 22,033 | 160,768,000 |

以上 MACs 是项目统一的 Conv/Linear 可复现下界。SLR 四相点积、加权 Value 与位置
加权另有 1,689,600 functional MACs，未混入上表，Norm/激活/Softmax 也未计入。

## 12. Forward/Backward/AMP/DDP

| 模型 | Forward | Backward | AMP | 三卡 DDP |
|---|---|---|---|---|
| PResNet18 + SLR | PASS | PASS | NOT RUN（本机无 CUDA） | NOT RUN（本机无三卡 CUDA） |
| HRNetV2-W18 + SLR | PASS | PASS | NOT RUN（本机无 CUDA） | NOT RUN（本机无三卡 CUDA） |

`tests.test_slr_neck` 共 13 项：12 PASS、1 项 CUDA AMP 跳过；HRNet 定向回归
12 PASS、1 项 CUDA 跳过。服务器验收命令：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/validate_slr_neck.py --amp-smoke
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9922 tools/validate_slr_neck.py --ddp-smoke --amp
```

DDP 工具对两个 candidate 均使用真实 SyncBN、`find_unused_parameters=False`，各执行
两次 forward/backward，以捕获第二步才出现的 unused parameter 问题。

## 13. 批量脚本

`tools/train_all_dut_modules_3gpu.sh` 现在固定只包含：

1. PResNet18 + SLR
2. HRNetV2-W18 + SLR

没有 `sort | shuf`，保留三卡、AMP、seed、端口递增、日志及 output 存在即 SKIP。

## 14. Dry-run

```text
Total configs: 2
Will run: 2
Will skip: 0
1. [RUN] configs/rtdetr/rtdetr_r18vd_dut_anti_uav_slr.yml
2. [RUN] configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_slr.yml
DRY RUN complete: no directory was created and no training was launched.
```

正式训练命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 bash tools/train_all_dut_modules_3gpu.sh
```
