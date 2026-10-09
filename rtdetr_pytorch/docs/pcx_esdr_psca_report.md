```text
========================================
PCX / ESDR / PSCA FINAL REPORT
========================================
```

本报告只记录实现与合成验证。没有读取 DUT-Anti-UAV 数据、没有执行优化器
step、没有启动正式训练，也没有生成或猜测测试集精度。

## 1. 修改文件

- 核心接入：`src/zoo/rtdetr/hybrid_encoder.py`、`src/core/yaml_config.py`
- 三个独立模块：`src/zoo/rtdetr/pcx_neck.py`、`esdr_neck.py`、
  `psca_neck.py`
- 六个候选 YAML：见第 14 节
- 队列与测试：`tools/train_pcx_esdr_psca_3gpu.sh`、
  `tools/test_pcx_esdr_psca_best.sh`、
  `tools/summarize_pcx_esdr_psca.py`
- 审计：`tools/validate_pcx_esdr_psca_necks.py`、
  `reports/pcx_esdr_psca_cpu.json`
- 测试：`tests/test_pcx_neck_unit.py`、`test_esdr_neck_unit.py`、
  `test_psca_neck_unit.py`、`test_pcx_esdr_psca_necks.py`、
  `test_train_pcx_esdr_psca_queue.py`、
  `test_pcx_esdr_psca_reporting.py`
- 兼容维护：旧 FDCR/RDCF 配置集合审计更新为合法的 14 个配置；
  output 清理白名单加入本轮六个实验。

两份正式 Baseline YAML 未修改，Git blob hash 仍为：

- PResNet18：`285d2b5a2b5027275b04332aa432ecb45855132d`
- HRNetV2-W18：`a0f3e7b9e625fe95aeedf56f2ddeea7018da4957`

## 2. PCX 实现位置

`PartialChannelCrossScaleExchange` 位于
`src/zoo/rtdetr/pcx_neck.py`。它连续切分 75% preservation / 25%
exchange，只让 exchange 子空间在 N3/N4/N5 相邻尺度间交换；每级使用独立
DW 3x3 + PW 1x1，最后可执行无参数 groups=2 channel shuffle。

## 3. ESDR 实现位置

`ExtremaSensitiveDownsample` 位于
`src/zoo/rtdetr/esdr_neck.py`，严格计算
`D_base + beta * Conv1x1(MaxPool2d(X))`。`beta` 是
`[1,C,1,1]` 的有界逐通道 LayerScale，实际初值 0.02；34/45 两处参数独立。

## 4. PSCA 实现位置

`PartialSpatialContextAttention` 位于
`src/zoo/rtdetr/psca_neck.py`。75% 通道保持，25% 通道执行单头空间注意力；
N3/N4 的 KV 池化 stride 分别为 4/2，QK 与 softmax 局部 FP32，AV 保持活动
精度，`alpha` 是逐通道有界 LayerScale。

## 5. 三个模块的准确插入位置

真实原路径为 `Backbone P3/P4/P5 -> input_proj -> AIFI -> top-down ->
bottom-up -> N3/N4/N5 -> Decoder`。

- PCX：完整原 CCFF 得到 N3/N4/N5 后，对三个输出统一处理。
- ESDR：只在两处原 `downsample_convs` 结果产生后、PAN concat 前补入极值残差。
- PSCA：完整原 CCFF 后只处理 N3/N4，N5 不调用任何 PSCA 运算。

三者只能单独启用；任意两者或三者同时启用会精确抛出：
`PCX, ESDR and PSCA must be evaluated independently.`。它们与 ACR、SLR、
PAF、BOR、DGFR、SPDR、FDCR、RDCF 也禁止混合。

## 6. Original disabled equivalence

PASS。开关全部关闭时不构造 PCX/ESDR/PSCA 参数，也不进入对应 forward。
PResNet18 和 HRNetV2-W18 均验证了同 seed 公共 state bit-exact，N3/N4/N5、
`pred_logits`、`pred_boxes` 在 `atol=1e-6, rtol=1e-5` 下等价。

## 7. 480/640/800 Forward

PASS。六个候选完整检测器均已在 CPU 对 480、640、800 三种正方形输入完成
forward，三层输出和预测张量均有限且 shape 正确。执行记录见
`reports/pcx_esdr_psca_cpu.json`。

## 8. Backward

PASS。六个候选均完成 encoder dummy backward 和完整检测器 dummy backward；
所有新增参数 `grad != None`，且梯度 finite、nonzero。没有执行 optimizer step。

## 9. AMP

- CPU BF16：六个候选 PASS，输出、loss、梯度均 finite。
- CUDA FP16 `--amp`：本地无 CUDA，未执行；服务器门禁命令：
  `CUDA_VISIBLE_DEVICES=1 python tools/validate_pcx_esdr_psca_necks.py --amp-smoke --output reports/pcx_esdr_psca_cuda_amp.json`

## 10. DDP

三卡 DDP smoke 入口已实现，但本地没有三张 CUDA 卡，未冒充执行结果。服务器
命令：

`CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9931 tools/validate_pcx_esdr_psca_necks.py --ddp-smoke --amp --output reports/pcx_esdr_psca_ddp.json`

该门禁逐个测试六配置的 startup、两次 forward/backward、梯度与无 unused
parameters/hang。

## 11. Optimizer 实际参数组

PASS。逐个读取真实 AdamW param groups：所有新参数均使用 main LR `3e-4`；
卷积 bias 使用 WD 0；卷积 weight、`raw_beta`、`raw_alpha` 使用 WD `1e-4`；
没有任何新参数进入 backbone LR `3e-5`。

## 12. Params / MACs

以下为 640 输入的实际执行计数；MAC 覆盖 Conv2d/Linear，PSCA 的函数式
QK/AV 另行计入。pool/interpolate/softmax/elementwise 等非乘加操作不在表中。

| Model | Params | Added params | Reported MACs | Added MACs |
|---|---:|---:|---:|---:|
| PResNet18 | 20,083,028 | - | 30,006,963,200 | - |
| PResNet18 + PCX | 20,171,284 | 88,256 | 30,119,936,000 | 112,972,800 |
| PResNet18 + ESDR | 20,215,124 | 132,096 | 30,138,035,200 | 131,072,000 |
| PResNet18 + PSCA | 20,107,988 | 24,960 | 30,368,230,400 | 361,267,200 |
| HRNetV2-W18 | 18,280,456 | - | 39,669,190,400 | - |
| HRNetV2-W18 + PCX | 18,368,712 | 88,256 | 39,782,163,200 | 112,972,800 |
| HRNetV2-W18 + ESDR | 18,412,552 | 132,096 | 39,800,262,400 | 131,072,000 |
| HRNetV2-W18 + PSCA | 18,305,416 | 24,960 | 40,030,457,600 | 361,267,200 |

PSCA attention 单独为 QK `102,400,000`、AV `204,800,000`，合计
`307,200,000` MACs；它已包含在上表 PSCA reported MACs 中，没有重复计算。

## 13. Fairness audit

PASS。六个 resolved candidate 与各自正式 Baseline 的差异只包含对应方法配置、
`output_dir` 和必要的 `__include__` 元数据。继承协议保持：200 epochs、
16/GPU、3 GPU/global 48、main LR `3e-4`、backbone LR `3e-5`、AdamW 与原
weight decay、5-epoch warmup + cosine、EMA、AMP、seed 0、训练 480-800
multi-scale、val/test 640，以及原 AIFI/Decoder/Matcher/Loss/queries/denoising。

## 14. 六个 YAML 路径

1. `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pcx.yml`
2. `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_pcx.yml`
3. `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_esdr.yml`
4. `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_esdr.yml`
5. `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_psca.yml`
6. `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_psca.yml`

## 15. dry-run

队列静态契约测试 PASS（固定六项顺序、仅 fresh、三张不同 GPU、AMP、seed、
skip/continue/final status）。本地 Windows 环境没有 Bash，真实 Bash dry-run 尚未
执行；服务器必须先运行：

`CUDA_VISIBLE_DEVICES=1,2,3 bash tools/train_pcx_esdr_psca_3gpu.sh --dry-run`

预期首行包含 `Total experiments = 6`，并按 PCX(PRes/HR)、ESDR(PRes/HR)、
PSCA(PRes/HR) 打印 config、output 与 FRESH 命令，且不创建输出目录。

## 16. 六实验连续训练状态

`NOT STARTED`。遵守停止条件，Codex 未启动正式训练。只有第 9、10、15 节的
服务器门禁全部 PASS 后，才执行：

`CUDA_VISIBLE_DEVICES=1,2,3 bash tools/train_pcx_esdr_psca_3gpu.sh`

脚本输出到 `output/three_gpu_b16_warmup_cosine/<experiment>`；目录已存在则
SKIP，一个实验失败会记录后继续下一个，队列结束后对失败返回非零。

## 17. 最终统一 Test 结果

训练尚未开始，所以 AP/AR 结果均为 `PENDING`，没有伪造数字。训练完成后运行：

`CUDA_VISIBLE_DEVICES=1 bash tools/test_pcx_esdr_psca_best.sh`

该脚本固定以 `best.pth`、`split=test`、640x640、EMA、FP32、workers=2 测试
两份已有 Baseline 和六个候选，并生成完整 12 项 COCO 指标及相对各自 backbone
的 AP/AP75/APS/ARS gain JSON/Markdown。

| Model | AP | AP50 | AP75 | APS | APM | APL | AR100 | ARS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PResNet18 | PENDING | | | | | | | |
| +PCX | PENDING | | | | | | | |
| +ESDR | PENDING | | | | | | | |
| +PSCA | PENDING | | | | | | | |
| HRNetV2-W18 | PENDING | | | | | | | |
| +PCX | PENDING | | | | | | | |
| +ESDR | PENDING | | | | | | | |
| +PSCA | PENDING | | | | | | | |

```text
PCX pluggable: YES
ESDR pluggable: YES
PSCA pluggable: YES

All disabled restore original model: YES

Works with PResNet18: YES
Works with HRNetV2-W18: YES

Training hyperparameters changed: NO
Multi-scale settings changed: NO

Six experiments ready: NO (server CUDA AMP, 3-GPU DDP and Bash dry-run gates pending)
```
