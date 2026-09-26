# Persistent Detail Relay（PDR）实现与验收说明

## 1. 实际主干审计

对当前工程 `PResNet18-d` 的真实执行路径使用 `1×3×640×640` 输入：

| 张量 | 原始 stage | 通道 | 空间尺寸 | stride |
|---|---|---:|---:|---:|
| C2 | `res_layers[0]` | 64 | 160×160 | 4 |
| C3 / P3 | `res_layers[1]` | 128 | 80×80 | 8 |
| C4 / P4 | `res_layers[2]` | 256 | 40×40 | 16 |
| C5 / P5 | `res_layers[3]` | 512 | 20×20 | 32 |

检测器仍只向 HybridEncoder 返回 P3/P4/P5，通道 `[128,256,512]`、stride
`[8,16,32]` 不变。没有把 YAML 中的说明当作结构依据，PDR 的主分支通道由
PResNet 构造时的 `_out_channels` 传入。

## 2. PDR 数据流

```text
C2 ── DetailMemory(1×1 Conv-BN-Act) ── D2(32, s4)
 │                                      │
 └─ original Stage3 ── M3              └─ PixelUnshuffle(2)
                         │                 1×1 Conv-BN-Act + DW3×3 ── D3(64, s8)
                         └─ M3 + α3·g3·Proj(D3) ── C3
                                      │                         │
                                      │                         └─ PixelUnshuffle(2)
                                      │                            1×1 + DW3×3 ── D4(96, s16)
                                      └─ original Stage4 ── M4             │
                                                               M4 + α4·g4·Proj(D4) ── C4
                                                                                 │
                                                                 original Stage5 ── C5
```

`D4` 只从 `D3` 生成，不从增强后的 C3 重新构造。PDR34 中原 Stage5 的输入是
增强后的 C4。主干原 stage、下采样路径和权重 key 均保留。

语义门控在局部 FP32 中计算：

```text
agreement = cosine(M, E)
g = rho + (1-rho) * sigmoid((agreement-theta)/tau)
output = M + alpha * g * E
alpha = alpha_max * sigmoid(raw_alpha)
```

默认 `rho=0.25`、`tau=0.2`、`theta=0`（可学习）、`alpha_max=0.5`，
`raw_alpha` 通过反 sigmoid 初始化，使实际 `alpha=0.1`，而不是把 raw 值误当
实际融合强度。NoGate 仅去掉 `g`，仍保留有界 alpha。

## 3. 模式与公平性

| 配置 | relay3/inject3 | relay4/inject4 | semantic gate |
|---|---|---|---|
| PDR3 | 开 | 关 | 开 |
| PDR34 | 开 | 开 | 开 |
| PDR34-NoGate | 开 | 开 | 关 |

三个配置只相对 DUT Baseline 增加 `PDR.*` 和独立 `output_dir`。它们继承同一份
三卡、每卡 batch=16、200 epoch、AdamW、WarmupCosineLR、数据增强、640 eval、
每轮验证、每 10 轮 checkpoint 和 best.pth 规则。MERT、SECD 及其他 backbone
实验保持关闭。`PDR.enabled=false` 时不构造任何 PDR module/parameter，也不消耗
额外随机数，走原始 forward。

## 4. 参数与计算量

默认通道下的新增参数拆分如下（包含 Conv weight、BN affine、alpha/theta；不把
BN running statistics 当作参数）：

| 部件 | PDR3 | PDR34 |
|---|---:|---:|
| DetailMemory | 2,112 | 2,112 |
| Relay3 | 9,024 | 9,024 |
| Projection3 + alpha3 + theta3 | 8,450 | 8,450 |
| Relay4 | 0 | 25,824 |
| Projection4 + alpha4 + theta4 | 0 | 25,090 |
| 合计 | 19,586 | 70,500 |

| 模型 | 完整检测器 Params | Δ Params | Profiler GFLOPs 下界 | Δ GFLOPs |
|---|---:|---:|---:|---:|
| Baseline | 20,083,028 | 0 | 61.1519 | 0 |
| PDR3 | 20,102,614 | +19,586 | 61.4755 | +0.3236 |
| PDR34 | 20,153,528 | +70,500 | 61.6364 | +0.4845 |
| PDR34-NoGate | 20,153,526 | +70,498 | 61.6351 | +0.4832 |

参数量由实际 Conv/BN/alpha/theta 结构精确统计。仅新增 Conv 使用 `2×MAC`
分别为 PDR3 `0.32195`、PDR34 `0.48200` GFLOPs；PixelUnshuffle 是重排。
表中的完整检测器 PyTorch profiler 结果还计入其支持的部分逐元素运算，但仍是
算子覆盖不完整的下界。运行下面命令可在
目标服务器的 PyTorch/CUDA 版本上输出统一口径的 Baseline/PDR 对比：

```bash
python tools/test_pdr.py --test-complexity
```

## 5. 预训练、AMP、DDP 与调试

加载原始 PResNet18 权重时使用严格白名单：只有 `pdr.*` 可以缺失，任何其他
missing/unexpected key 都会报错；原始 backbone 交集 key 必须逐元素保持一致。
标准 PResNet18-d state_dict 的原始 138 个 tensor key 均应 matched；PDR3 新增
loader missing key 全部来自 `pdr.*`（PDR3 为 22 个，PDR34 为 39 个；BN 的
`num_batches_tracked` 由 PyTorch 兼容逻辑处理），`unexpected=0`。
AMP 只在 cosine 门控局部转 FP32，其他分支继续使用 autocast。DDP 不需要
`find_unused_parameters=True`，每种模式只构造实际启用的 relay/injection。

`debug: true` 时可读取 `model.backbone.pdr.last_debug_stats` 和
`last_debug_tensors`。包括 `raw_alpha3/4`、`alpha3/4_eff`、`theta3/4`、
`gate3/4_mean/std/min/max`、detail/main/injection norm，以及 gate/agreement/energy
空间图。它们保留为 device tensor，不做每 batch `.item()`、CPU 拷贝或打印。

checkpoint 参数审计：

```bash
python tools/test_pdr.py --checkpoint output/rtdetr_r18vd_dut_anti_uav_pdr34/best.pth
```

## 6. 验收命令

```bash
python tools/test_pdr.py --test-baseline --test-shape --test-grad --test-pretrained --test-amp
torchrun --nproc_per_node=2 --master_port=9918 tools/test_pdr.py --test-ddp --model pdr3
torchrun --nproc_per_node=2 --master_port=9919 tools/test_pdr.py --test-ddp --model pdr34
torchrun --nproc_per_node=2 --master_port=9920 tools/test_pdr.py --test-ddp --model pdr34_nogate
```

测试覆盖：Baseline 精确等价、640 shape、PixelUnshuffle 偶数尺寸保护、D2→D3→D4
链路、Stage5 输入增强 C4、所有核心参数梯度、门控范围、alpha 初值、AMP finite、
原始预训练 key、配置注册表不串扰和双进程 DDP。

## 7. 三卡训练命令

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9909 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9910 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr3.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9911 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr34.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9912 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr34_nogate.yml --amp --seed 0
```

## 8. 本次实际验收结果

| 项目 | 结果 | 证据 |
|---|---|---|
| Baseline equivalence | PASS | PDR disabled，固定 seed/eval/input，三个输出逐元素相等（`rtol=atol=0`） |
| Shape / persistent path | PASS | 640 输入为 128×80²、256×40²、512×20²；D4 输入等于 D3；Stage4/5 分别读取增强 C3/C4 |
| Gradient | PASS | DetailMemory、Relay3/4、Projection3/4、theta3/4、raw_alpha3/4 梯度均 finite 且非零 |
| Gate / alpha | PASS | gate 位于 `[0.25,1]`，两个 effective alpha 初值均为 0.1 |
| Pretrained | PASS | matched=138，PDR34 missing=39 且全部为 `pdr.*`，unexpected=0 |
| AMP | PASS | PDR3、PDR34、NoGate CUDA autocast forward/backward 均 finite |
| DDP | PASS | 三模式分别通过 2-rank Gloo + `find_unused_parameters=False` forward/backward |
| YAML 公平性 | PASS | 原 DUT 审计 0 failure；候选只改变 PDR/output_dir |

这里只执行了合成输入验收和复杂度前向，没有启动数据集训练。

## 9. 修改文件

| 路径 | 类型 | 用途 |
|---|---|---|
| `src/nn/backbone/backbone_modules/pdr.py` | 新增 | 四个 PDR 核心组件、持续 relay、门控、融合与 debug |
| `src/nn/backbone/backbone_modules/__init__.py` | 修改 | 导出 PDR 组件 |
| `src/nn/backbone/presnet.py` | 修改 | 可选挂载、独立 forward、预训练严格白名单 |
| `src/core/yaml_config.py` | 修改 | PDR 默认关闭及全局配置防串扰 |
| `configs/rtdetr/include/pdr.yml` | 新增 | 唯一默认 PDR 参数模板 |
| `configs/rtdetr/*pdr*.yml` | 新增（3 个） | PDR3、PDR34、PDR34-NoGate 独立实验 |
| `tests/test_pdr.py` | 新增 | 单元/集成/AMP/预训练验收 |
| `tools/test_pdr.py` | 新增 | 测试、复杂度、DDP、checkpoint 参数检查入口 |
| `tools/analyze_dut_models.py` | 修改 | 公平性审计识别独立 PDR 字段 |
