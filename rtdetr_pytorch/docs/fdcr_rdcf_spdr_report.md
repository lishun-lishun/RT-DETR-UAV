# FDCR / RDCF / SPDR FINAL REPORT

```text
========================================
FDCR / RDCF / SPDR FINAL REPORT
========================================
```

生成日期：2026-10-07。工程根目录：`rtdetr_pytorch/`。

本报告只记录已经实际执行的检查。当前 Windows 工作区没有服务器端
`output/three_gpu_b16_warmup_cosine/`、Linux Bash 或 CUDA，因此服务器
checkpoint、CUDA AMP、三卡 DDP、正式训练和真实 test 指标均明确标为待执行。

## 1. LPRU 删除清单

已删除两份实验 YAML、LPRU 类与 import、HybridEncoder 中的配置解析/构造/调用、
全局 YAML 注册、旧训练/测试/验证入口、旧单元测试、旧说明和旧 CPU 报告。SPDR
从共用的 `resample_neck.py` 中独立保留，结构、beta、PixelUnshuffle、projection
和优化器协议未修改。

本地没有服务器结果目录，所以没有虚报删除服务器 checkpoint。服务器同步代码后，
先检查：

```bash
find output/three_gpu_b16_warmup_cosine -maxdepth 1 -type d -iname '*lpru*' -print
```

确认仅命中后，删除以下两个明确目标（这是本任务唯一尚待服务器执行的删除动作）：

```text
output/three_gpu_b16_warmup_cosine/rtdetr_r18vd_dut_anti_uav_lpru/
output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav_lpru/
```

## 2. LPRU 源码是否完全移除

PASS。对 `src/ configs/ tools/ tests/` 搜索
`LPRU|LearnablePixelReassembly|PixelReassembly` 无匹配。历史说明不会参与运行。

## 3. PResNet18 Baseline 是否正常

PASS（CPU synthetic）。真实 YAML 构建、完整 detector 前向、关闭态等价和
640×640 复杂度前向均通过；参数量为 20,083,028。

## 4. HRNetV2-W18 是否正常

PASS（CPU synthetic，审计时仅关闭 ImageNet 权重下载）。真实 YAML 构建、完整
detector 前向、关闭态等价和 640×640 复杂度前向均通过；参数量为 18,280,456。

## 5. SPDR 是否正常

PASS（CPU synthetic）。PResNet18+SPDR 和 HRNetV2-W18+SPDR 均真实构建并完成
128×128 完整 detector 前向；SPDR 单元公式、零 beta、反向传播和输入契约通过。

## 6. HRNet-SPDR resume checkpoint

PENDING SERVER AUDIT。本地没有服务器 output。严格检查命令：

```bash
python tools/inspect_resume_checkpoint.py output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav_spdr --json-out output/hrnet_spdr_resume_audit.json
```

检查器依次选择：完整 `last.pth`、内部保存 epoch 最大的完整
`checkpoint*.pth`（包含滚动 `checkpoint.pth`）、最后才是完整 `best.pth`。

## 7. HRNet-SPDR resume epoch

PENDING SERVER AUDIT。只有 checkpoint 内的 `last_epoch`/兼容 `epoch` 值有效，
不根据文件名猜测。`last_epoch >= 199` 才视为 200 epoch 已完成，否则从
`last_epoch + 1` 续训。

## 8. Optimizer state loaded

PENDING SERVER AUDIT。严格检查器要求非空 optimizer state；缺失时输出
`STRICT RESUME NOT POSSIBLE`，队列不会将其伪装成续训。

## 9. Scheduler state loaded

PENDING SERVER AUDIT。严格检查器要求 `lr_scheduler`/兼容旧 scheduler key。

## 10. EMA loaded

PENDING SERVER AUDIT。严格检查器要求非空 EMA state；统一最终测试额外要求
checkpoint 中存在可用的 `ema.module`。

## 11. FDCR 源码位置

`src/zoo/rtdetr/fdcr_neck.py`。实现同尺度 AvgPool 3×3 高频/低频分解、低频
DWConv 5×5、高频 DWConv 3×3、各自 1×1 projection、concat 后 1×1 fusion，
以及 `gamma_max * tanh(raw_gamma)` 的 per-channel LayerScale。无新增 Norm、
Attention、Gate、Softmax 或 Sigmoid。

## 12. RDCF 源码位置

`src/zoo/rtdetr/rdcf_neck.py`。训练态为 DWConv 3×3、1×9、9×1 直接相加，
再接 1×1 projection 和 `eta_max * tanh(raw_eta)`。`switch_to_deploy()` 将三个
kernel/bias 精确融合为单个带 bias 的 depthwise 9×9，并删除训练分支。

## 13. FDCR 准确插入位置

`HybridEncoder` 完整执行原始 AIFI、top-down FPN 和 bottom-up PAN/CCFF，得到
`[N3,N4,N5]` 后，两个不共享参数的 `fdcr3/fdcr4` 只处理 N3/N4。N5 不进入
FDCR，Decoder 接收 `[FDCR3(N3), FDCR4(N4), N5]`。

## 14. RDCF 准确插入位置

与 FDCR 相同，严格位于完整原始 CCFF 之后；独立 `rdcf3/rdcf4` 只处理 N3/N4，
N5 原样透传。FDCR 与 RDCF 同时开启会直接抛出：
`FDCR and RDCF must be evaluated independently.`；两者也禁止与 SPDR、ACR、
SLR、PAF、BOR、DGFR 混用。

## 15. FDCR disabled equivalence

PASS。PResNet18 与 HRNet 两条完整 detector 路径均验证：state dict/common seeded
weights bit-exact；backbone、N3/N4/N5、`pred_logits`、`pred_boxes` 在
`atol=1e-6, rtol=1e-5` 下等价；关闭时不构造 FDCR 参数或执行 FDCR 调用。

## 16. RDCF disabled equivalence

PASS。与第 15 节相同，两种 Backbone 的完整 detector 均通过；关闭时不构造
RDCF 参数或执行 RDCF 调用。

## 17. RDCF deploy reparameterization equivalence

PASS。PResNet18 与 HRNet 两种结构均满足训练态/转换部署态
`atol=1e-5, rtol=1e-4`；N5 bit-exact；转换后不存在三个训练分支，仅存在
`reparam_conv`，且转换后的 state dict 可严格加载到 `deploy: true` 模型。

## 18. 480/640/800 forward

PASS（CPU FP32）。四个新候选均完成完整 detector 的 480、640、800 输入前向，
N3/N4/N5 分辨率分别为输入的 1/8、1/16、1/32，输出为
`pred_logits=[1,300,1]`、`pred_boxes=[1,300,4]`，无 NaN/Inf。

## 19. Backward

PASS（CPU FP32）。四候选均执行 Encoder backward 和完整 detector training
forward 的递归 dummy loss backward；所有 FDCR/RDCF 新增参数均
`grad is not None`、finite 且 nonzero。未执行 optimizer step 或正式训练。

## 20. AMP

CPU BF16 autocast：PASS，无 NaN/Inf，新增参数梯度存在且有限。

CUDA FP16 AMP：PENDING SERVER。命令：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/validate_fdcr_rdcf_necks.py --amp-smoke
```

## 21. 3-GPU DDP

PENDING SERVER。已提供严格 world-size=3、NCCL、两步 forward/backward、
`find_unused_parameters=False` 的入口：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9926 tools/validate_fdcr_rdcf_necks.py --ddp-smoke --amp
```

## 22. Optimizer 实际 LR / weight decay

PASS。四候选的全部新增参数都属于 HybridEncoder/main optimizer：

- 所有新增参数 LR = `3e-4`，没有落入 backbone LR `3e-5`。
- Conv bias：weight decay = `0`。
- Conv weight、`raw_gamma`、`raw_eta`：weight decay = `1e-4`。

逐参数名称、optimizer group、LR、weight decay 已记录在
`reports/fdcr_rdcf_cpu.json`。

## 23. Params / MACs

以下 MAC 为 640×640 实际执行的 Conv2d/Linear MAC lower bound，不包含 pooling、
norm、插值、激活、concat、elementwise 等操作：

| Model | Params | Conv/Linear MACs | Delta Params | Delta MACs |
|---|---:|---:|---:|---:|
| PResNet18 Original | 20,083,028 | 30,006,963,200 | - | - |
| PResNet18 + FDCR | 20,627,796 | 32,173,747,200 | +544,768 | +2,166,784,000 |
| PResNet18 + RDCF train | 20,230,484 | 30,586,547,200 | +147,456 | +579,584,000 |
| PResNet18 + RDCF deploy | 20,257,108 | 30,697,139,200 | +174,080 | +690,176,000 |
| HRNetV2-W18 Original | 18,280,456 | 39,669,190,400 | - | - |
| HRNetV2-W18 + FDCR | 18,825,224 | 41,835,974,400 | +544,768 | +2,166,784,000 |
| HRNetV2-W18 + RDCF train | 18,427,912 | 40,248,774,400 | +147,456 | +579,584,000 |
| HRNetV2-W18 + RDCF deploy | 18,454,536 | 40,359,366,400 | +174,080 | +690,176,000 |

注意：三个方向 kernel 合并为稠密 9×9 后，部署形态的理论参数/MAC 略高于三条
稀疏训练分支；其价值是单分支执行和精确等价，不应误报成 MAC 降低。

## 24. Resolved config fairness audit

PASS。四对比较均只存在 `__include__` metadata、`output_dir` 和对应 FDCR/RDCF
namespace 的合法差异。epoch、batch/global batch、LR、weight decay、optimizer、
scheduler、warmup、EMA、multi-scale、增强、decoder、matcher、loss、queries 等
协议未改变。FDCR 固定 `gamma_max=0.30, gamma_init=0.05`；RDCF 固定
`eta_max=0.30, eta_init=0.05, deploy=false`。

## 25. 四个新 YAML 完整路径

- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_fdcr.yml`
- `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_fdcr.yml`
- `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_rdcf.yml`
- `configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_rdcf.yml`

## 26. 五实验 dry-run 顺序

脚本静态契约与 14 项 resume/queue 单测 PASS；Windows 本机没有 Linux Bash 且
没有服务器 checkpoint，因此真实 dry-run 仍为 PENDING SERVER：

```bash
bash tools/train_fdcr_rdcf_spdr_3gpu.sh --dry-run
```

固定顺序为：

1. HRNetV2-W18 + SPDR `[RESUME]`
2. PResNet18 + FDCR `[FRESH]`
3. HRNetV2-W18 + FDCR `[FRESH]`
4. PResNet18 + RDCF `[FRESH]`
5. HRNetV2-W18 + RDCF `[FRESH]`

只有严格 checkpoint 审计和服务器 dry-run 正常后，才运行：

```bash
bash tools/train_fdcr_rdcf_spdr_3gpu.sh
```

## 27. 五实验训练状态

NOT STARTED。本次没有启动任何正式训练。队列固定三卡
`CUDA_VISIBLE_DEVICES=1,2,3`、`nproc_per_node=3`、AMP、seed 0；SPDR 强制
`-r <latest_full_checkpoint>`，其余四项绝不传 `-r`。单项失败会记录 return code
和最后完整 checkpoint，并继续下一项。

## 28. 最终统一 Test 结果

PENDING TRAINING。测试脚本固定串行比较七项：两条正式 Baseline、PRes FDCR、
PRes RDCF、HR FDCR、HR RDCF、HR SPDR；统一 `best.pth`、`split=test`、GPU 1、
workers 2、EMA required、FP32、640×640。单项失败继续，最终汇总完整 12 项 COCO
指标并自动计算 AP/AP75/APS/ARS 相对对应 Backbone 基线的 gain：

```bash
bash tools/test_fdcr_rdcf_spdr_best.sh
```

输出 `summary.json/.csv/.md` 以及
`fdcr_rdcf_spdr_comparison.json/.md`。当前没有正式模型结果，因此不填造 AP。

## Final status

```text
LPRU fully removed: YES (repository); SERVER OUTPUT NOT VERIFIED
FDCR pluggable: YES
RDCF pluggable: YES
FDCR disabled restores original: YES
RDCF disabled restores original: YES
RDCF train/deploy equivalent: YES
HRNet-SPDR strict resume: PENDING SERVER AUDIT
Training hyperparameters changed: NO
Multi-scale settings changed: NO
Five experiments scheduled: YES
```

CPU 审计原始记录：`reports/fdcr_rdcf_cpu.json`。
