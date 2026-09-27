# ACR-Neck 实现与验收报告

ACR-Neck（Agreement-Calibrated Residual Routing Neck）已作为原
`HybridEncoder` 的可选跨尺度路由接入。它没有复制或替换整个
`HybridEncoder`；`ACR.enabled: false` 时不构造任何 ACR 模块或参数，仍执行原始
CCFF 语句。

## A. 修改文件

| 文件 | 类型 | 用途 |
|---|---|---|
| `src/zoo/rtdetr/acr_neck.py` | 新增 | Energy Calibration、Semantic Agreement、Scale-Exclusive Residual 及组合单元 |
| `src/zoo/rtdetr/hybrid_encoder.py` | 修改 | 在原 Top-down/PAN 融合点可选调用 ACR |
| `src/core/yaml_config.py` | 修改 | 注入默认关闭配置并防止不同 YAML 间共享配置泄漏 |
| `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_acr.yml` | 新增 | PResNet18 + ACR 公平实验 |
| `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_acr.yml` | 新增 | HRNetV2-W18 + ACR 公平实验 |
| `tools/train_all_dut_modules_3gpu.sh` | 修改 | 将上述两项放在原队列最前面，保留原 skip/三卡/AMP 机制 |
| `tools/validate_acr_neck.py` | 新增 | 等价性、形状、梯度、优化器、AMP、DDP、复杂度与测速验收 |
| `tests/test_acr_neck.py` | 新增 | ACR 自动化回归测试 |
| `reports/acr_validation_cpu.json` | 新增 | 本地 CPU 实测结果 |

## B. 真实连接位置

当前调用链为：P3/P4/P5 → `HybridEncoder.input_proj` → P5 的 AIFI →
Top-down FPN → Bottom-up PAN → 三尺度输出。ACR 位于 `input_proj` 之后，连接为：

- P5→P4：`acr_54` 对上采样深层特征做能量校准及语义一致性门控，浅层 P4 完整保留；
- P4→P3：`acr_43` 以相同方式只门控深层注入，浅层 P3 完整保留；
- P3→P4：`acr_43` 将 P3 与校准后深层特征的显著残差独立下采样，按 `beta34` 注入原 PAN 分支；
- P4→P5：`acr_54` 将对应残差按 `beta45` 注入原 PAN 分支。

原 `CSPRepLayer`、AIFI、Decoder、Matcher、Loss、P3/P4/P5 层级和
`feat_strides=[8,16,32]` 均未修改。

## C. Baseline equivalence

`ACR=None` 与 `ACR.enabled=false` 在相同随机种子、相同权重和相同输入下，所有
`state_dict` 键及三层输出均通过 `atol=1e-6, rtol=1e-5` 比较：**PASS**。

## D. Shape

输入 `1×3×640×640`：

| 模型 | HybridEncoder 输出 | Detector 输出 |
|---|---|---|
| PResNet18 + ACR | `[1,256,80,80]`, `[1,256,40,40]`, `[1,256,20,20]` | logits `[1,300,1]`, boxes `[1,300,4]` |
| HRNetV2-W18 + ACR | `[1,256,80,80]`, `[1,256,40,40]`, `[1,256,20,20]` | logits `[1,300,1]`, boxes `[1,300,4]` |

两项均为 **PASS**，没有新增 P2/P6 或额外输出层。

## E. Params/MACs

以下为 `1×3×640×640` 实测。MACs 沿用项目现有 Conv/Linear 下界口径，不计
Norm、插值、逐元素统计/门控、Softmax 等 functional 运算；不能视为完整硬件 FLOPs。

| 模型 | Whole Params | Whole Conv/Linear MACs | ACR 新增 Params | ACR 新增 Conv/Linear MACs |
|---|---:|---:|---:|---:|
| PResNet18 + Original | 20,083,028 | 30,006,963,200 | – | – |
| PResNet18 + ACR | 21,263,702 | 31,186,611,200 | 1,180,674 | 1,179,648,000 |
| HRNetV2-W18 + Original | 18,280,456 | 39,669,190,400 | – | – |
| HRNetV2-W18 + ACR | 19,461,130 | 40,848,838,400 | 1,180,674 | 1,179,648,000 |

## F. Optimizer

两个 ACR 配置的实际分组一致：

- `raw_beta34/raw_beta45`：主学习率 `3e-4`，weight decay `1e-4`；
- `detail_downsample.conv.weight`：主学习率 `3e-4`，weight decay `1e-4`；
- `detail_downsample.norm.weight/bias`：主学习率 `3e-4`，weight decay `0`。

因此 ACR 未误入 backbone 的 `3e-5` 组，Norm 规则与原 Encoder 一致。

## G. 测试状态与命令

本地 CPU 已完成：Forward **PASS**、Backward **PASS**、关闭等价 **PASS**、
配置公平性 **PASS**、优化器分组 **PASS**。本机 PyTorch 为 CPU 版，因此 CUDA AMP
与三卡 NCCL DDP 明确记为 **NOT RUN**，不能伪报通过；可在训练服务器执行：

```bash
python tools/validate_acr_neck.py --smoke --complexity --output reports/acr_validation_server.json
CUDA_VISIBLE_DEVICES=1 python tools/validate_acr_neck.py --benchmark --amp --warmup 50 --iterations 100 --output reports/acr_benchmark_amp.json
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9921 tools/validate_acr_neck.py --ddp-smoke --amp
```

`python -m unittest tests.test_acr_neck -v` 的结果为 10 项中 9 项通过、1 项仅因本机
无 CUDA 跳过；没有失败项。项目全量旧测试中仍存在与本次 ACR 无关的既有失败，
主要是历史公平性测试仍要求旧 LR/Loader 设置以及测试间 selective import 污染；本次没有
为通过这些旧断言而修改现行训练协议。

## H. 批量训练计划

`bash tools/train_all_dut_modules_3gpu.sh --dry-run` 已通过语法和计划检查。本地无
对应 output 时，队列前两项为：

1. `[RUN] configs/rtdetr/rtdetr_r18vd_dut_anti_uav_acr.yml` →
   `output/three_gpu_b16_warmup_cosine/rtdetr_r18vd_dut_anti_uav_acr`
2. `[RUN] configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_acr.yml` →
   `output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav_acr`
3. 随后接原有队列。

服务器正式执行仍为三卡、AMP、相同 seed，并沿用 output 存在即 SKIP：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/train_all_dut_modules_3gpu.sh
```

两份新 YAML 只继承各自原实验并覆盖 `output_dir` 和 `ACR`；200 epochs、每卡
batch 16、global batch 48、主 LR `3e-4`、backbone LR `3e-5`、480–800
multi-scale、Val/Test 640、EMA、AMP、seed、数据、Decoder/Matcher/Loss 均保持不变。
