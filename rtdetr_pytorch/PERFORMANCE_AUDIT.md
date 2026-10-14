# RT-DETR-R18VD 计算复杂度与推理性能专项审计

审计日期：2026-10-14  
审计对象：`configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml` 对应的 RT-DETR-R18VD  
固定输入：`N×3×640×640`  
约束：未修改模型结构、训练/测试配置或 checkpoint；未进行训练。

## 1. 结论摘要

1. **`112.007 GFLOPs` 不应作为该模型的标准单次前向计算量。** 当前模型在 `1×3×640×640` 下，独立 Conv/Linear hook 为 `30.006963 GMACs = 60.013926 GFLOPs（2×MACs）`；CPU-dispatch `torch.profiler` 为 `61.151875 GFLOPs`；fvcore 为 `30.469274 G`（fvcore 约定 FMA=1）；THOP 为 `30.206234 GMACs`。多种独立方法均聚集在 `60～61.2 GFLOPs（2×MAC口径）`，没有证据表明模型实际计算量翻倍。
2. 原性能脚本固定输入正确，也没有在 FLOPs 统计区域显式执行两次模型前向。但是它同时启用 `CPU` 和 `CUDA` profiler activity，随后无来源过滤地累加所有 `event.flops`。`112.007` 很可能是当前 PyTorch/CUPTI 组合下的 profiler activity 聚合异常或口径混合值；新审计脚本会分别统计 CPU-dispatch-only 与 CPU+CUDA，直接给出二者差值。不能在取得该分解前武断认定为模型重复计算。
3. `60.014 GFLOPs` 是 **Conv2d/Linear 的 2×MACs**，对应 `30.007 GMACs`，不是另一个模型的结果。它不含 `grid_sample`、归一化、激活、池化、top-k 等，也不完整覆盖函数式 attention；因此比完整的 `61.15 GFLOPs` 略低是正常的。
4. SFA-DETR 论文的 `56.9 GFLOPs` 没有公开计算工具及 MAC/FLOP 换算规则，不能确认与本审计完全同口径。其参数量也是 `19.87M`，与当前严格枚举的 `20.083028M` 不同，说明至少实现或参数统计边界并非完全一致。
5. 当前 `46.53 FPS` 与 `21.491 ms` 严格互相吻合：`1000 / 21.491 = 46.53`。现有计时已包含预热、`torch.inference_mode()`、CUDA Event 和计时前后同步，未发现明显的计时逻辑错误。
6. 论文 `119 FPS` 是 RTX 3090 上的结果，论文表 1 同时给出 **Test Batch Size=16**；当前主指标是 A30、batch=1。论文没有说明精度、PyTorch/CUDA、预热、同步以及是否包含前后处理，因此两者不能直接比较或据此判断当前实现存在性能故障。

## 2. 已知测试协议与复现边界

| 项目 | 当前结果 |
|---|---:|
| GPU | NVIDIA A30 24 GB |
| NVIDIA driver | 535.261.03（来自已有服务器环境记录） |
| PyTorch | 2.0.1+cu118（来自已有服务器环境记录） |
| PyTorch CUDA runtime | 11.8 |
| 输入 | `1×3×640×640` |
| 精度 | CUDA FP16 autocast |
| 模式 | `model.eval()` + `torch.inference_mode()` |
| 计时范围 | PyTorch model forward only |
| 不计入 | 文件读取、解码、resize/normalize、H2D、RT-DETR postprocessor |
| warmup / timed iterations | 50 / 200（性能脚本默认值） |
| 参数量 | 20,083,028 |
| Conv/Linear | 30.006963 GMACs / 60.013926 GFLOPs（2×MACs） |
| 原 mixed-activity profiler | 112.007 GFLOPs（不建议作为论文复杂度值） |
| batch=1 | 46.53 images/s；21.491 ms/image |
| batch=8 | 必须在 A30 上运行新增审计脚本后填写，本文不臆测 |

checkpoint 中的数值不会改变固定网络图的参数量和理论 MACs；本地交叉验证用相同解析配置构建网络，参数总数与用户提供的 checkpoint 结果精确一致。A30 的 batch=1/batch=8 速度只能以服务器实测为准。

## 3. 原 FLOPs 脚本核查

### 3.1 输入和前向次数

- 输入常量是 `INPUT_SHAPE = (1, 3, 640, 640)`，符合要求。
- profiler 区域只调用一次 `model(image)`，未发现脚本层面的双前向。
- 模型顶层 forward 依次且各执行一次 `backbone → encoder → decoder`，未发现重复提取完整 backbone/encoder 特征。
- 推理态下 `multi_scale` 不生效，审计脚本又显式设为 `None`，不会暗中改变 640 输入。
- `eval_spatial_size` 只决定缓存 anchor 的空间形状，不会再 resize 输入。

### 3.2 原 profiler 口径问题

原脚本同时采集：

```text
ProfilerActivity.CPU + ProfilerActivity.CUDA
```

然后对 `profile.key_averages()` 中所有具有 `flops` 的事件直接求和。PyTorch 官方文档说明 `with_flops=True` 只对特定算子按输入形状套用公式，重点是矩阵乘和二维卷积，并不是完整模型的统一理论 FLOPs 计数器：[PyTorch profiler 文档](https://docs.pytorch.org/docs/stable/autograd.html)。

审计脚本为此增加两次相互独立的采集：

1. `CPU activity only`：统计 ATen dispatch 公式，作为推荐的 profiler 值；
2. `CPU + CUDA activities`：仅用于诊断当前 CUPTI 环境是否把 device activity 的 FLOPs 再次计入。

同时检查“具有 FLOPs 的父事件和子事件”是否嵌套，以排除简单的层级重复相加。本地同图 CPU 审计中，该潜在嵌套重复量为 `0`。

### 3.3 算子覆盖边界

原 Conv/Linear hook 正确计算：

```text
Conv2d MACs = output elements × (Cin/groups) × Kh × Kw
Linear MACs = output elements × in_features
```

再乘以 2 得到“乘法和加法各算一次”的 GFLOPs。它明确遗漏：

- Multihead/self-attention 中的函数式矩阵运算；
- deformable attention 的 `grid_sample` 和插值；
- LayerNorm/BatchNorm、激活、softmax、sigmoid；
- pooling、top-k、gather、repeat、concat 和逐元素运算。

因此 `60.014` 是主要 Conv/Linear 工作量的可靠下界，不是完整算子覆盖值。

## 4. 分模块参数量与计算量

分区定义：

- Backbone：`model.backbone`；
- Hybrid Encoder：`model.encoder`；
- Detection Head：decoder 内的 `enc_output`、encoder class/bbox head、decoder class/bbox heads；
- Decoder：`model.decoder` 除上述 head 外的输入投影、query 选择/位置投影与 transformer decoder layers。

### 4.1 严格参数枚举与 Conv/Linear MACs

| 模块 | Params | Conv/Linear GMACs | Conv/Linear GFLOPs（2×MACs） |
|---|---:|---:|---:|
| Backbone | 11,199,968 | 16.865690 | 33.731379 |
| Hybrid Encoder | 4,965,120 | 8.441037 | 16.882074 |
| Decoder（不含 Detection Head） | 3,320,160 | 2.919014 | 5.838029 |
| Detection Head | 597,780 | 1.781222 | 3.562445 |
| **Total** | **20,083,028** | **30.006963** | **60.013926** |

这张表的四行互斥，总和与模型严格参数枚举及全模型 Conv/Linear hook 完全一致，不存在模块间重复求和。

### 4.2 fvcore 交叉验证

fvcore 把一个 fused multiply-add 计为一个 flop，因此其数值更接近 GMAC，而不是上表的 `2×MACs`。fvcore 支持按模块归属聚合，但仍需检查 unsupported operators；这是其官方设计：[fvcore FLOP count 文档](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md?plain=1)。

| 模块 | fvcore G（FMA=1） |
|---|---:|
| Backbone | 16.922624 |
| Hybrid Encoder | 8.580915 |
| Decoder（不含 Detection Head） | 3.173760 |
| Detection Head | 1.791974 |
| **Total** | **30.469274** |

fvcore 未支持或未赋公式的算子及调用次数：

```text
aten::max_pool2d: 1
aten::add: 49
aten::avg_pool2d: 3
aten::div: 10
aten::mul: 56
aten::scaled_dot_product_attention: 4
aten::gelu: 1
aten::silu_: 28
aten::topk: 1
aten::repeat: 2
aten::sigmoid: 4
aten::softmax: 3
aten::sub: 3
aten::sum: 3
aten::rsub: 3
aten::log: 3
```

因此 fvcore 的 `30.469274 G` 仍然不是“所有实际 GPU 指令”的绝对真值。

### 4.3 THOP 交叉验证

本地使用当前环境兼容的 THOP 得到：

```text
30.206234 GMACs
60.412467 GFLOPs（人为按 2×MACs 换算）
```

THOP 本质上以 forward/module hook 规则统计 MACs；不受支持的函数式/custom operator 可能缺失。其官方说明也明确将输出称为 MACs，并指出 FLOPs 常近似为 2×MACs：[THOP 文档](https://github.com/ultralytics/thop/blob/main/README.md)。本次旧版 THOP 返回的参数数是 `19,029,330`，低于 PyTorch 严格枚举，因此 **THOP 参数量不采用**；该差异本身也证明不同工具不能只看最终一个数字。

### 4.4 三种结果的关系

| 方法 | 原生输出口径 | 结果 |
|---|---|---:|
| 自定义 Conv/Linear hook | MACs | 30.006963 GMACs |
| 自定义 Conv/Linear hook | 2×MACs | 60.013926 GFLOPs |
| THOP | MACs | 30.206234 GMACs |
| fvcore | FMA=1，含部分非 Conv/Linear op | 30.469274 G |
| torch.profiler CPU-dispatch | 已支持 ATen op 的公式 FLOPs | 61.151875 GFLOPs |
| 原 CPU+CUDA mixed activity | 无来源过滤的事件 FLOPs 总和 | 112.007 GFLOPs |

前五项相互吻合；最后一项明显脱离交叉验证区间，应视为 profiler 聚合诊断值，而不是模型标准 GFLOPs。

## 5. FP16、同步、预热和关闭梯度核查

原性能脚本的计时实现符合基本规范：

- `--amp` 后使用 `torch.autocast(device_type='cuda', dtype=torch.float16)` 包围每次 forward；
- 模型处于 `eval()`；
- 预热默认 50 次；
- 全程使用 `torch.inference_mode()`，不建立 autograd graph；
- 计时前 `torch.cuda.synchronize()`；
- 每次使用 CUDA Event 记录起止；
- 全部事件提交后再次同步，再读取 elapsed time；
- 输入已提前放到 GPU，不把 H2D 计入模型 forward；
- FPS 使用 `1000 / mean_ms`，batch=1 下定义正确。

新增审计脚本不只检查“是否传入 `--amp`”，还会记录：

- autocast context 内 `torch.is_autocast_enabled()`；
- 所有 Conv/Linear 输出 dtype 计数；
- 最终 `pred_logits`/`pred_boxes` dtype；
- `verified_fp16_execution` 布尔值。

这能区分“命令带了 `--amp`”和“关键层确实产出 FP16 tensor”。

## 6. batch=1 与 batch=8 性能协议

| Batch | batch latency | Throughput | 平均每图耗时 | 状态 |
|---:|---:|---:|---:|---|
| 1 | 21.491 ms | 46.53 images/s | 21.491 ms/image | 已有 A30 实测，实时主指标 |
| 8 | 待脚本输出 | 待脚本输出 | `batch latency / 8` | 必须在同一 A30、同一进程实测 |

batch=8 的 `images/s` 是吞吐率，不是单路实时延迟。论文采用 Test Batch Size=16，所以其 `119 FPS` 更可能属于批处理吞吐条件；除非作者公开其精确测速代码，否则不能把它解释为 `8.403 ms` 的 batch=1 单图延迟。

## 7. 推理路径和瓶颈检查

### 7.1 未发现的问题

- 顶层 forward 没有 `.item()`、`.cpu()`、`.numpy()` 或显式 CUDA synchronize；未发现 forward 内强制 CPU-GPU 同步。
- Backbone、Hybrid Encoder、Decoder 在顶层各调用一次；未发现完整特征的重复计算。
- 当前参数总数与用户报告严格一致，说明审计的是同一 R18VD 结构而非误加载的更大模型。

### 7.2 值得关注但不等于实现错误的开销

- decoder 的多尺度 deformable attention 是纯 PyTorch `F.grid_sample` 循环实现，而不是单一融合 CUDA extension；理论 FLOPs 很低的 sampling/reshape/permute 仍可能产生明显 kernel launch 与显存访问开销。
- eager PyTorch 中包含多次小 Linear、LayerNorm、softmax、top-k、gather、repeat/permute；这些算子不一定是 FLOPs 主体，却会限制 batch=1 延迟。
- Hybrid Encoder 的 RepVggBlock 在当前未部署形态执行 `3×3 + 1×1` 两条分支。这是源码设计，不是意外重复整个特征网络；部署融合可能改变实际速度/计算图，但本审计遵守要求，没有转换模型。
- `torch.profiler` 自身有显著开销，只用于算子排序，不能用 profiler 包围区间的时间计算 FPS。

新增脚本会输出 batch=1 的 Backbone、Hybrid Encoder、Decoder（含 head）CUDA Event 分段时间，以及 top-30 CUDA operator、自定义同步/传输标记，供 A30 上定位实际瓶颈。

## 8. 与 SFA-DETR 论文的可比性

SFA-DETR 原文第 4.2 节说明硬件为 **RTX 3090 24 GB + Intel Xeon E5-2698 v4**；表 1 给出 **Image Size=640、Test Batch Size=16**；表 6/8 报告 RT-DETR 为 `19.87M / 56.9 GFLOPs / 119 FPS`：[SFA-DETR 论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12610626/)。

可以比较：

- 两者均以 640 输入为目标；
- 参数量和理论复杂度的数量级；
- 在明确说明工具与 MAC/FLOP 约定后，同一套本地脚本测得的相对复杂度。

不能直接比较：

- A30 batch=1 的 46.53 FPS 与 RTX 3090 test batch=16 的 119 FPS；
- 不同 PyTorch/CUDA/cuDNN、不同 eager/fused/export backend 的 FPS；
- 未知精度（论文没有写 FP32/FP16）、未知预热/同步和未知计时范围；
- 当前 `20.083M` 模型与论文 `19.87M` 模型的绝对 FLOPs，二者至少统计边界或具体实现不同；
- 原脚本 `112.007` mixed-activity profiler 与论文未公开工具得到的 `56.9`。

## 9. 明确回答

### 当前 112.007 GFLOPs 是否合理？

**作为标准单次模型前向 GFLOPs，不合理；作为特定 mixed-activity profiler 的事件求和值，可以出现，但不能用于论文表格。** 独立 Conv/Linear、THOP、fvcore 和 CPU-dispatch profiler 均将该模型定位在约 `60～61.2 GFLOPs（2×MACs）`。没有证据表明模型真实计算量翻倍。

### 为什么 Conv/Linear 统计只有 60.014 GFLOPs？

它实际是 `30.007 GMACs × 2`，只覆盖执行到的 Conv2d/Linear。attention 矩阵运算和部分逐元素运算使 profiler 增加约 `1.14G`；`grid_sample`、softmax、归一化等仍没有被完整计入。所以 `60.014` 与完整交叉验证值接近但略低，属于预期结果。

### 与论文 56.9 GFLOPs 是否属于同一统计口径？

**无法证明属于完全相同口径。** 论文未披露 FLOPs 工具和 FMA 换算规则，且模型参数量不同。两者都明显接近 RT-DETR-R18 的“约 30 GMACs / 约 60 GFLOPs”量级，但不能直接用 `60.014 - 56.9` 解释为某个模块增加了固定计算量。

### 46.53 FPS 是否存在明显性能问题？

**从现有证据不能认定存在明显实现或计时错误。** 它与 21.491 ms 完全一致，计时包含正确预热、同步、CUDA Event、inference mode 和 FP16 autocast。是否还有优化空间，应看新增脚本在 A30 上给出的 stage timing、top CUDA operators 及 batch=8 吞吐，不能仅凭另一张 GPU 上的论文 FPS 判定。

### 如果不存在实现问题，与论文 119 FPS 不同的主要因素是什么？

主要是：**RTX 3090 与 A30 硬件不同；论文 test batch=16 而当前实时主指标 batch=1；论文精度、软件版本、预热同步、计时范围和部署后端未披露；两边模型参数量也不完全相同。** 另外，当前 eager PyTorch 的纯 `grid_sample` deformable attention 与多次小算子对 batch=1 延迟不友好，而理论 GFLOPs 无法表达这些访存和 kernel-launch 成本。

## 10. A30 复现命令

若服务器尚未安装交叉验证工具：

```bash
python -m pip install fvcore thop
```

在 `rtdetr_pytorch` 目录运行（物理 GPU 1 会映射成进程内的 `cuda:0`）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/audit_rtdetr_performance.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml -r output/rtdetr_r18vd_dut_anti_uav/best.pth --device cuda:0 --amp --batches 1 8 --warmup 50 --iterations 200 --profile-iterations 10 --output output/rtdetr_r18vd_dut_anti_uav/performance_audit_results.json
```

输出 JSON 会记录 GPU、driver、PyTorch、CUDA、cuDNN、输入、精度、计时边界、AMP dtype 验证、两种 batch 速度、分阶段耗时、三套复杂度结果、unsupported operators 和 profiler bottleneck 表。
