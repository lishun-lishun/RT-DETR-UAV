# ==============================
# DGFR FINAL REPORT
# ==============================

本报告对应当前 DUT-Anti-UAV 工程中的第一版 **DGFR-Neck（Direct Global Fusion Residual Neck）**。报告只记录已经由本地 CPU 验证得到的结果；本地环境没有 CUDA，因此 CUDA AMP、三卡 DDP、正式 200 轮训练和最终 Test 指标均明确标记为尚未运行。

## 1. 新增/修改文件

新增文件：

- `src/zoo/rtdetr/dgfr_neck.py`
- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_dgfr.yml`
- `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr.yml`
- `tests/test_dgfr_neck.py`
- `tests/test_dgfr_neck_unit.py`
- `tools/validate_dgfr_neck.py`
- `reports/dgfr_validation_cpu.json`
- `docs/dgfr_neck.md`

修改文件：

- `src/zoo/rtdetr/hybrid_encoder.py`：注册并调用可选 DGFR 支路。
- `src/core/yaml_config.py`：加入 `DGFR` 默认关闭、类型检查和全局配置原子覆盖。
- `tools/train_all_dut_modules_3gpu.sh`：训练队列固定为两个 DGFR 实验。
- `tests/test_slr_neck.py`：正式 DUT 配置集合断言同步加入两份 DGFR YAML。

原始 PResNet18、HRNetV2-W18、RT-DETR Decoder、Matcher、Loss 和两份 Baseline YAML 均未修改。

## 2. DGFR 源码位置

DGFR 实现在 `src/zoo/rtdetr/dgfr_neck.py`：

- `DirectScaleAdapter`：每个“来源尺度 -> 目标尺度”的独立对齐支路。
- `DirectGlobalFusion`：一个目标尺度的三来源直接融合专家。
- `DGFRNeck`：三个尺度专家、逐通道有界 LayerScale 和最终残差注入。

每个目标尺度都直接接收 `X3/X4/X5`，九条尺度对齐路径不共享参数。每个来源先变换为 64 channels，拼接得到 192 channels，再交给该尺度独立的 `CSPRepLayer(num_blocks=1, expansion=0.5)` 输出 256 channels。

## 3. DGFR 准确插入位置

实际数据流为：

```text
Backbone P3/P4/P5
  -> input_proj
  -> X3/X4/X5
  -> AIFI（X5 在这里被更新）
  -> 原始 Top-down CCFF
  -> 原始 Bottom-up CCFF
  -> O3/O4/O5

同时：AIFI 后的 X3/X4/X5
  -> DGFR3/DGFR4/DGFR5
  -> E3/E4/E5

最终：Yl = Ol + Gamma_l * El
```

调用位于 `HybridEncoder.forward()` 原始 `outs` 全部生成之后：

```python
if self.dgfr_enabled:
    outs = self.dgfr(proj_feats, outs)
```

DGFR 只使用 `/8、/16、/32` 三层，不读取 HRNet 的 `/4` 分支，不增加 P2/P6，也不增加 Decoder level。

## 4. Original CCFF 是否保持完整

**YES。**

原有 `input_proj`、AIFI、`lateral_convs`、`fpn_blocks`、`downsample_convs` 和 `pan_blocks` 的构造及 forward 顺序均保留。DGFR 没有删除、替换或改写 Top-down/Bottom-up CCFF，只在原始 `O3/O4/O5` 完成后执行并行残差注入。

## 5. DGFR disabled Baseline equivalence

本地 CPU 对 PResNet18 和 HRNetV2-W18 分别验证了 `DGFR.enabled=false`：

| 对照 | state_dict | Backbone 输出 | O3/O4/O5 | pred_logits | pred_boxes |
|---|---|---|---|---|---|
| PResNet18 | EXACT | PASS | PASS | PASS | PASS |
| HRNetV2-W18 | EXACT | PASS | PASS | PASS | PASS |

比较阈值为 `atol=1e-6, rtol=1e-5`。关闭时不会实例化 `encoder.dgfr`，所以没有隐藏参数或隐藏计算。使用相同随机种子开启 DGFR 后，全部原模型同名权重也保持 tensor 精确一致；DGFR 初始化使用隔离 RNG，不会改变原 Encoder/Decoder 的初始化序列。

## 6. 输入输出 shape

DGFR 输入是 `input_proj` 后且已经完成 AIFI 更新的三层统一特征：

```text
X3: [B, 256, H/8,  W/8]
X4: [B, 256, H/16, W/16]
X5: [B, 256, H/32, W/32]
```

三个专家分别生成 `E3/E4/E5`，输出保持：

```text
Y3: [B, 256, H/8,  W/8]
Y4: [B, 256, H/16, W/16]
Y5: [B, 256, H/32, W/32]
```

640 输入时，PResNet18 和 HRNetV2-W18 的最终 Encoder 输出均为：

```text
[B, 256, 80, 80]
[B, 256, 40, 40]
[B, 256, 20, 20]
```

完整 Detector 输出保持 `pred_logits=[1,300,1]`、`pred_boxes=[1,300,4]`。

## 7. 480/640/800 forward 结果

| Backbone | 输入 | Backbone P3/P4/P5 channels | DGFR 后 Encoder 空间尺寸 | 结果 |
|---|---:|---|---|---|
| PResNet18 | 480 | 128/256/512 | 60×60, 30×30, 15×15 | PASS |
| PResNet18 | 640 | 128/256/512 | 80×80, 40×40, 20×20 | PASS |
| PResNet18 | 800 | 128/256/512 | 100×100, 50×50, 25×25 | PASS |
| HRNetV2-W18 | 480 | 36/72/144 | 60×60, 30×30, 15×15 | PASS |
| HRNetV2-W18 | 640 | 36/72/144 | 80×80, 40×40, 20×20 | PASS |
| HRNetV2-W18 | 800 | 36/72/144 | 100×100, 50×50, 25×25 | PASS |

所有结果均为有限值。上采样使用 `size=目标特征.shape[-2:]` 的 nearest interpolation；下采样只使用原项目风格的 3×3、stride=2、padding=1 `ConvNormLayer`，没有 Pooling 或 stride=4 卷积。

## 8. Backward 结果

**PResNet18+DGFR：PASS。HRNetV2-W18+DGFR：PASS。**

合成 forward/backward 验证覆盖全部九个 adapter、三个独立 fusion block 及 `raw_gamma3/4/5`。两种 Backbone 下所有 78 个 DGFR 可训练参数 tensor 都获得了存在、有限且非零的梯度。

逐通道 Gamma 的有效初值实测为 `0.049999997`，与目标 `0.05` 一致：

```text
Gamma_l = 0.25 * tanh(raw_gamma_l)
Gamma_l shape = [1, 256, 1, 1]
有效范围 = [-0.25, +0.25]
```

调试统计 `Gamma mean/min/max`、`||E_l||/||O_l||` 和 `Y_l norm` 均已验证为有限且 detached；`debug=false` 时不记录，也不会逐 step 打印。

DGFR 测试命令的本地结果：

```text
Ran 21 tests in 9.750s
OK (skipped=1)
```

即 20 项 CPU 测试通过，1 项 CUDA AMP 测试因本地无 CUDA 跳过。
包含既有 PAF/BOR、SLR 和 HRNet 的联合回归也已执行：共 64 项，60 项通过、4 项 CUDA 测试因本地无 CUDA 跳过、0 项失败。

## 9. AMP 结果

**NOT RUN：本地 CPU 环境没有 CUDA，需在服务器执行验证。**

不能将跳过的 CUDA 测试报告为 PASS。服务器单卡 CUDA AMP 验证命令：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/validate_dgfr_neck.py --amp-smoke
```

该命令会依次对 PResNet18+DGFR、HRNetV2-W18+DGFR 执行完整模型 FP16 autocast forward/backward，并检查输出、loss 和所有 DGFR 梯度是否存在 NaN/Inf。

## 10. 三卡 DDP 结果

**NOT RUN：本地 CPU 环境没有三张 CUDA GPU，需在服务器执行验证。**

服务器三卡 DDP + AMP 验证命令：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9924 tools/validate_dgfr_neck.py --ddp-smoke --amp
```

验证工具要求真实 `WORLD_SIZE=3`，使用 NCCL、SyncBatchNorm 和 DDP；两个候选各执行两次 forward/backward，并检查 DGFR 梯度、unused parameter 风险和进程同步。只有命令明确输出两个候选均 `PASS` 后，才能将本节改为 PASS。

## 11. DGFR optimizer 实际 LR/weight_decay

两种 Backbone 的真实 AdamW 参数组结果一致：

| DGFR 参数类型 | 参数 tensor 数 | LR | weight_decay |
|---|---:|---:|---:|
| Conv/CSP weights + `raw_gamma3/4/5` | 28 | 3e-4 | 1e-4 |
| Norm weights/biases | 50 | 3e-4 | 0 |

DGFR 共 `1,321,472` 个参数，全部属于 `encoder.dgfr.*` 并进入 Neck/Encoder 的 main LR `3e-4`，没有任何参数误入 backbone LR `3e-5`。三个 `raw_gamma` 均使用 `lr=3e-4, weight_decay=1e-4`。

现有 HRNet Backbone 的 BatchNorm 参数规则保持原样，本次没有借 DGFR 实验修改全局优化器分组，确保与既有 HRNet 对照结果可比。

## 12. Params/MACs（640）

| 模型 | Whole Params | Whole Conv/Linear MACs | ΔParams | ΔConv/Linear MACs |
|---|---:|---:|---:|---:|
| PResNet18 Original | 20,083,028 | 30,006,963,200 | — | — |
| PResNet18 + DGFR | 21,404,500 | 32,990,489,600 | +1,321,472 | +2,983,526,400 |
| HRNetV2-W18 Original | 18,280,456 | 39,669,190,400 | — | — |
| HRNetV2-W18 + DGFR | 19,601,928 | 42,652,716,800 | +1,321,472 | +2,983,526,400 |

MACs 是统一可复现的 Conv/Linear lower bound；没有计入 BatchNorm、插值、激活、拼接、LayerScale 和逐元素残差运算。计算结果已写入 `reports/dgfr_validation_cpu.json`。未因计算量较大而擅自减小 `hidden_dim`、`fusion_channels` 或改动训练协议。

## 13. Resolved config 公平性结果

**PResNet18 对照：PASS。HRNetV2-W18 对照：PASS。**

两份候选相对各自正式 Baseline 的 resolved config 只存在以下合法差异：

- `DGFR.enabled/fusion_channels/gamma_max/gamma_init`
- `output_dir`
- `__include__` 元数据

以下真实训练协议均未变化：

- 200 epochs；每卡 batch 16；3 GPUs；global batch 48。
- AdamW：main LR `3e-4`，backbone LR `3e-5`，betas `[0.9,0.999]`，默认 weight decay `1e-4`。
- WarmupCosine：5 epochs warmup，start factor `0.1`，minimum ratio `0.01`。
- EMA：decay `0.9999`，warmups `667`；gradient clip `0.1`。
- 训练多尺度：`[480,512,544,576,608,640,640,640,672,704,736,768,800]`。
- Validation/Test resize：640×640；训练增强、DataLoader、Dataset split 不变。
- Decoder 3 层、300 queries、100 denoising queries；AIFI、Matcher、Loss 不变。
- AMP 由训练命令 `--amp` 开启；seed 为 0。
- PResNet18 和 HRNetV2-W18 均保持各自原 ImageNet pretrained 行为。

## 14. 两个新增 YAML 路径

```text
configs/rtdetr/rtdetr_r18vd_dut_anti_uav_dgfr.yml
configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr.yml
```

第一版参数固定为：

```yaml
DGFR:
  enabled: true
  fusion_channels: 64
  gamma_max: 0.25
  gamma_init: 0.05
```

未进行自动超参数搜索，也没有混入 ACR、SLR、PAF、BOR 或其他实验模块。

## 15. dry-run 结果

已执行：

```bash
bash tools/train_all_dut_modules_3gpu.sh --dry-run
```

实际结果：

```text
Total configs: 2
Will run: 2
Will skip: 0
1. [RUN] configs/rtdetr/rtdetr_r18vd_dut_anti_uav_dgfr.yml
2. [RUN] configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr.yml
DRY RUN complete: no directory was created and no training was launched.
```

脚本没有 `sort | shuf`，固定先 PResNet18+DGFR、再 HRNetV2-W18+DGFR。最终 output 目录只要已经存在就会 `SKIP`，不会删除或覆盖旧实验。

## 16. 正式训练命令/状态

**状态：NOT STARTED。本地没有启动任何 200 epoch 正式训练。**

正式训练前必须先在服务器完成第 9、10 节 AMP/DDP smoke，并确认均为 PASS。随后在项目根目录运行：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 bash tools/train_all_dut_modules_3gpu.sh
```

脚本内部固定使用三卡 `torchrun --nproc_per_node=3`、`--amp --seed 0`，输出到：

```text
output/three_gpu_b16_warmup_cosine/rtdetr_r18vd_dut_anti_uav_dgfr
output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr
```

训练过程每轮在 val split 验证，按 val AP@[0.50:0.95] 保存 `best.pth`，每 10 个 completed epochs 额外保存编号 checkpoint。

## 17. 最终 Test 指标

**状态：暂无。正式训练尚未启动，因此没有 DGFR `best.pth`，也没有可报告的 Test 指标。**

不能使用 val 指标代替 test。训练完成后必须使用 `tools/test_dut.py --split test`、640×640、FP32 和 checkpoint 中的 EMA 权重。

一次性顺序测试两个 DGFR 模型：

```bash
for name in rtdetr_r18vd_dut_anti_uav_dgfr rtdetr_hrnetv2_w18_dut_anti_uav_dgfr; do OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/test_dut.py -c "configs/rtdetr/${name}.yml" -r "output/three_gpu_b16_warmup_cosine/${name}/best.pth" --split test --num-workers 2 --output-dir "output/three_gpu_b16_warmup_cosine/${name}/test_eval"; done
```

分别测试：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/test_dut.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_dgfr.yml -r output/three_gpu_b16_warmup_cosine/rtdetr_r18vd_dut_anti_uav_dgfr/best.pth --split test --num-workers 2 --output-dir output/three_gpu_b16_warmup_cosine/rtdetr_r18vd_dut_anti_uav_dgfr/test_eval
```

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/test_dut.py -c configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr.yml -r output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr/best.pth --split test --num-workers 2 --output-dir output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr/test_eval
```

待测试完成后填写：

| Model | AP | AP50 | AP75 | APS | APM | APL | AR100 | ARS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PResNet18 | existing（本轮未重测） | — | — | — | — | — | — | — |
| PResNet18 + DGFR | pending | pending | pending | pending | pending | pending | pending | pending |
| HRNetV2-W18 | existing（本轮未重测） | — | — | — | — | — | — | — |
| HRNetV2-W18 + DGFR | pending | pending | pending | pending | pending | pending | pending | pending |

只有取得同一 test split、同一 640×640 FP32 协议下的数值后，才能计算 DGFR 对 PResNet18 和 HRNetV2-W18 的真实增益。

---

```text
DGFR pluggable:
YES

DGFR disabled restores original RT-DETR:
YES

Works with PResNet18:
YES

Works with HRNetV2-W18:
YES

Training hyperparameters changed:
NO

Only two new experiments:
YES
```
