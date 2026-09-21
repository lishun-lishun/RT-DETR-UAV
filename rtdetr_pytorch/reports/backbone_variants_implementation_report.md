# RT-DETR-R18 Backbone 结构实验：HSDR / PHSB 实现报告

## 结论与范围

已实现四个独立 YAML 模式：Baseline、HSDR-B、HSDR-A、PHSB。所有模式只输出 P3/P4/P5，通道为 128/256/512，stride 为 8/16/32。未修改 HybridEncoder、AIFI、CCFF、Query Selection、Decoder、Matcher、Detection Loss、DN、数据管线和训练协议；没有实现 HSDR+PHSB、MERT 联合或任何其他算子。

## 1—3：原始 R18 的真实结构

在 `src/nn/backbone/presnet.py` 中，`PResNet` 用 `BasicBlock` 构建 R18，`ResNet_cfg[18]` 为 `[2,2,2,2]`；Stem 为三个 3×3 ConvNormLayer（第一层 stride 2）再接 3×3 stride 2 max-pool。实际 `1×3×640×640` 前向、逐 stage hook 结果：

| stage | 实现位置 | block 数 | 输出 shape | channel | stride | 是否给 Encoder |
|---|---|---:|---|---:|---:|---|
| S2 | `res_layers[0]` | 2 | `[1,64,160,160]` | 64 | 4 | 否 |
| S3 | `res_layers[1]` | 2 | `[1,128,80,80]` | 128 | 8 | P3 |
| S4 | `res_layers[2]` | 2 | `[1,256,40,40]` | 256 | 16 | P4 |
| S5 | `res_layers[3]` | 2 | `[1,512,20,20]` | 512 | 32 | P5 |

S3→S4 的降采样发生于 S4 第一个 `BasicBlock`；原 forward 顺序执行 `res_layers`，返回 `return_idx=[1,2,3]` 的三个 feature。原预训练由 `torch.hub.load_state_dict_from_url` 加载；原 key 例如 `conv1.conv1_1.conv.weight`、`res_layers.1.blocks.0.*`。YAML 通过 `__include__` 和 `PResNet.__share__` 注入配置；当前 R18 基础配置实际使用 `depth:18`，并不是按文件名推测。

## 4—8：HSDR

4. HSDR-B：`[2,3,2,1]`，总 8 个 BasicBlock。
5. HSDR-A：`[2,4,3,1]`，总 10 个 BasicBlock。
6. 构造时先以原 `[2,2,2,2]` 建立全部原生 `BasicBlock`，然后在 forked RNG 流中向 S3/S4 末尾追加原生 stride-1、shortcut=true block，并删除 S5 最后一个 block。这样已存在的 stage key、初始权重和后续 Encoder/Decoder RNG 顺序不被打乱。stage channel、stride、原始 block 类型完全不变。
7. 预训练使用 `strict=False` 并对 missing/unexpected 作精确白名单校验：只允许新增 S3/S4 block 缺失，以及被删除的 `res_layers.3.blocks.1.*` 为 unexpected；Stem 或原 block 缺失会报错。加载后记录 `pretrained_load_report` 并输出数量及缺失/忽略 key。
8. 新 block 使用与原 `BasicBlock` 完全相同的构造和默认 PyTorch 初始化，没有复制旧 block 或添加特殊层。

## 9—17：PHSB

9. PHSB 插在原 S3 完成后，额外建一个保持 stride 8 的并行语义分支；原 S2/S3/S4/S5 不替换。
10. 输入是原 `F3_base = res_layers[1](F2)`，不是 P2、Stem、原图或已经增强的 P3。
11. `Ch = max(8, round(C3×branch_ratio/8)×8)`。真实 `C3=128`、`branch_ratio=.75`，因此 reduction channel = 96，未硬编码 96。
12. 分支为 `1×1 reduction + 3 个原生 BasicBlock(stride=1) + 两个 projection`；所有 HR block 保持 80×80。
13. `E3 = ConvNorm(1×1)(H3)`，`P3 = F3_base + α3 E3`。
14. `E4 = ConvNorm(3×3,stride=2)(H3)`，`P4 = F4_base + α4 E4`。
15. 原 Stage4 严格读取 `F3_base`，不读取 `P3`。该点有 pre-hook 与手工复算测试。
16. 原 Stage5 读取已经增强的 `P4`，故 `P5=Stage5(P4)`；有 pre-hook 与手工复算测试。
17. 两个独立 raw alpha 参数，`αi=0.20×tanh(raw_alphai)`，默认均初始化 0。PHSB debug 可选打印两路 alpha、E3/E4 与 base 的范数及比例、HR activation mean/std；正常 `debug=false` 不执行标量 CPU 同步。

## 18—19：等价性

18. 将修改前、只读 `git show HEAD` 的 PResNet 与本次 Baseline 使用相同 seed、模型权重、640 输入和 eval 模式比较：P3/P4/P5、Encoder 输出、Decoder 输出、最终 logits/boxes 全部 `atol=rtol=0` 逐元素一致。
19. PHSB 的 `α3=α4=0` 时，P3/P4/P5 及完整检测器预测也与 Baseline `atol=rtol=0` 逐元素一致。

## 20—25：复杂度、延迟与显存

本地 RTX 5060 Ti / PyTorch 2.10.0+cu128；完整检测器 `batch=1`、640×640、AMP、100 次预热、500 次 `torch.cuda.Event` 计时。峰值显存为该检测器推理中 `torch.cuda.max_memory_allocated`，不是训练显存。

| Method | Params | ΔParams | 计数 GFLOPs | ΔGFLOPs | Mean ms | Median ms | P95 ms | FPS | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 20,083,028 | 0 | 112.0066 | 0 | 10.5519 | 10.4021 | 11.2131 | 94.77 | 158.50 |
| HSDR-B | 15,657,812 | −4,425,216 | 112.0072 | +0.0006 | 10.5126 | 10.3396 | 11.0462 | 95.12 | 134.83 |
| HSDR-A | 17,133,908 | −2,949,120 | 127.1079 | +15.1013 | 10.8515 | 10.7022 | 11.2550 | 92.15 | 141.88 |
| PHSB | 20,828,566 | +745,538 | 126.7946 | +14.7880 | 11.3822 | 11.2257 | 12.0069 | 87.86 | 164.91 |

GFLOPs 是 PyTorch profiler 已计数算子的**下界**，不是精确总 FLOPs；未分配 FLOPs 的 elementwise/fused 算子没有被擅自补估。HSDR-B 的结果值得注意：S3 128 channel/80×80 和 S5 512 channel/20×20 的单个 3×3 block 主要卷积 MAC 恰好接近（空间面积 16 倍、通道平方 1/16），因此这次真实计算量几乎相同，而不是预设的“必然明显增加”。相同 block 数仍然不能作为一般意义上的相同 FLOPs 证明。PHSB 计数 GFLOPs 相对 Baseline +13.2%，低于 40% 警戒线。本机速度不代表服务器 A30，应在服务器用相同脚本复测。

## 26：预训练与 state_dict

通过 mock 原 PResNet18 状态字典，精确验证原 key 载入结果（计数仅限 Backbone；`num_batches_tracked` 的 missing 行为由 PyTorch BatchNorm 处理）：

| Method | 加载的原 key | missing | unexpected |
|---|---:|---:|---:|
| Baseline | 138 | 0 | 0 |
| HSDR-B | 126 | 10（仅 S3 block 2） | 12（仅被删 S5 block 1） |
| HSDR-A | 126 | 30（仅 S3 blocks 2/3、S4 block 2） | 12（仅被删 S5 block 1） |
| PHSB | 138 | 47（仅 `phsb.*`） | 0 |

人为删除一个 Stem 权重时 HSDR 会抛异常，故不是无条件 `strict=False`。真实服务器 URL 预训练文件未在本地沙盒实际下载；服务器首次构建时应核对打印的报告。

## 27—30：梯度、AMP、数值

HSDR-B、HSDR-A、PHSB 各自执行 forward/dummy loss/backward，新增 block/branch 参数均有 finite 梯度。PHSB 三步梯度范数（alpha3/alpha4、reduction、HR block、P3 projection、P4 projection）如下：

| Step | α3 | α4 | reduction | HR block | P3 projection | P4 projection |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.009705 | 0.009500 | 0 | 0 | 0 | 0 |
| 1 | 0.003340 | 0.002974 | 8.065e−5 | 1.782e−4 | 2.145e−5 | 8.173e−5 |
| 2 | 0.001012 | 0.000918 | 9.792e−5 | 1.473e−4 | 1.681e−5 | 1.008e−4 |

alpha=0 时 branch 参数第一步零梯度是公式决定的正常现象；alpha 第一步有梯度，后续 branch 参数能学习。四种模式均在 CUDA AMP 下完成 640×640 Backbone forward/backward，loss、输出、输入梯度及所有已产生的参数梯度均 finite，无 dtype mismatch、NaN 或 Inf。完整检测器的 640×640 AMP inference 也通过基准脚本执行。

## 31—35：文件与配置

31. 新增 Python：`src/nn/backbone/phsb.py`、`tests/test_backbone_variants.py`、`tools/benchmark_backbone_variants.py`。
32. 修改原文件：`src/nn/backbone/presnet.py`、`tools/analyze_dut_models.py`、`tests/test_analyze_dut_models.py`。Baseline YAML 增加 `BackboneVariant.type: baseline`。
33. PResNet 修改用于 stage 深度重分配、PHSB 挂载及预训练 key 白名单；独立模块文件放置分支实现；旧公平性审计只接受相对官方配置多出的默认 Baseline 选择器，而仍禁止旧 MERT/SECD 方法改变 BackboneVariant；测试与基准工具用于可复现验收。
34. 新增 YAML：`rtdetr_r18vd_dut_anti_uav_hsdr_b.yml`、`rtdetr_r18vd_dut_anti_uav_hsdr_a.yml`、`rtdetr_r18vd_dut_anti_uav_phsb.yml`。
35. 相对 Baseline，HSDR-B/A 只改变 `output_dir` 和 `BackboneVariant.{type,stage_blocks,debug}`；PHSB 只改变 `output_dir`、`BackboneVariant.type` 和 `PHSB.{branch_ratio,num_blocks,alpha_init,alpha_max,debug}`。配置解析审计通过；MERT、SECD、CCED、GRER 与原有插件全部关闭。数据、Resize、multi-scale、batch16/GPU、200 epoch、每10 epoch checkpoint、验证每轮、Optimizer、LR、Scheduler、EMA、pretrained、Encoder、Decoder、Loss、Matcher 均继承 Baseline；随机种子由相同 CLI `--seed 0` 控制。

## 36：推荐训练顺序及命令

在服务器的 `rtdetr_pytorch` 目录执行；已有相同协议的 Baseline 可不重训。每个实验使用独立 output_dir，运行前确认对应目录没有旧 checkpoint 混杂。

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_hsdr_b.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_phsb.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_hsdr_a.yml --amp --seed 0
```

## 37：推荐评估与复测命令

真正的 DUT held-out **test** split 使用项目现有 `tools/test_dut.py`，其评估为 FP32；`tools/train.py --test-only` 实际评估的是 **val** split，不应误称 test。以下每条单独运行：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/test_dut.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_hsdr_b.yml -r output/rtdetr_r18vd_dut_anti_uav_hsdr_b/best.pth --split test
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/test_dut.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_phsb.yml -r output/rtdetr_r18vd_dut_anti_uav_phsb/best.pth --split test
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/test_dut.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_hsdr_a.yml -r output/rtdetr_r18vd_dut_anti_uav_hsdr_a/best.pth --split test
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_backbone_variants.py --amp --warmup 100 --iterations 500 --output reports/backbone_variants_benchmark_a30.json
```

验证功能与公平性审计：`python -m unittest discover -s tests -p test_backbone_variants.py -v`。最终比较记录 AP、AP50、AP75、APS/APM/APL、AR1/AR10/AR100、ARS/ARM/ARL，重点 AP/AP75/APS/ARS。

## 38：当前风险与限制

1. 本地完成了结构、等价性、模拟预训练映射、AMP/梯度、推理基准和旧路径回归；**没有运行200轮训练，因此不存在已验证的 AP 提升**。本地 torchvision 0.25 缺失旧项目所需 `torchvision.datapoints`，无法在此环境执行完整数据训练/验证；服务器应使用原可运行环境。
2. PHSB 两个 raw alpha 初始为零；尽管三步测试确认可学习，实际 AdamW 的 Backbone LR 为 1e−5，需观察训练过程中 alpha 是否长期太小。不要在第一轮未经实验改 LR/alpha。
3. PHSB 两路可影响 P3/P4，P4 再影响 P5；但更高分辨率上的额外 BN/BasicBlock 可能增加训练时激活显存与时间。本表只报告 **推理峰值显存**，不能据此推断 batch16 的训练显存。
4. HSDR-B 参数显著减少且计算量近似不变，可能因削弱 S5 造成大目标指标下降；HSDR-A 增加高分辨率计算，需检查 AP75/APS/ARS 是否真正受益，不能以 AP +0.1 的波动认定成功。
5. 官方 PResNet18 URL 权重在本地沙盒没有实际下载，预训练兼容性验证使用相同结构的模拟 state_dict；服务器实跑会显示 loaded/missing/unexpected 报告。如有非预期 Stem/旧 block key 缺失，代码会拒绝加载。
6. FLOPs 数字为 profiler 计数下界，且 CUDA profiler 在本地设备提示 CUPTI 活动缺失，但 CPU 侧 operator FLOP 计数和 CUDA Event 计时均正常；请用 A30 服务器复测。
7. debug 模式会打印 tensor 标量并同步 GPU，仅用于诊断；正式训练保持 `debug: false`。

本地相关回归测试共 79/79 通过（新方案11项、原 Backbone 插件13项、SECD23项、CCED/GRER16项、旧配置公平性审计16项）。
