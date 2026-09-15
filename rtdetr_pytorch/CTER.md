# CTER 实现与验收报告

本次以 `configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh.yml` 为唯一训练基础。
不修改原 MERT 源文件、该 YAML、PResNet 或检测网络结构，不重新引入旧实验模块。
CTER 为零参数、训练期辅助损失；CTER 负责跨阶段目标/背景可分性，MERT 继续负责双视图解码框细化轨迹一致性。

## 1. 当前 MERT 审计

以下以本次回退后的实际源码为准，不采用以前实验说明中的默认值。

| 环节 | 实际位置与行为 |
| --- | --- |
| 主实现 | `src/solver/mert.py`：`MicroShiftPairGenerator`、`RefinementTrajectoryEquivarianceLoss`、`MERTTrainingPlugin` |
| micro-shift | `_sample_shifts`，每张图从 `[-1,0,1]` 的非 `(0,0)` 组合中采样 |
| shifted image | `shift_images`，非循环平移，空白处填 0，保留既有 baseline fill 行为 |
| shifted GT | `shift_target`，归一化 cxcywh 转像素 xyxy，平移、裁边、转回归一化 cxcywh |
| 实例身份 | `shift_target` 通过 `torch.arange(num_gt)` 生成 `origin_gt_id`，裁边后保留对应 ID |
| 多尺度时机 | `MERTTrainingPlugin.prepare` 先按原 RT-DETR 多尺度策略缩放，再生成像素平移；forward 时临时禁用内部二次缩放 |
| 双视图 forward | late_xywh 是 `concat`：原图与 shifted 图拼成 `2B`，一次顶层 model forward，并非两次 model 调用 |
| Hungarian Matching | 轨迹损失 `forward` 中，对两视图各自的最后一层、非 DN 检测输出单独匹配；matcher 实现在 `src/zoo/rtdetr/matcher.py` |
| trajectory | `decoder_box_trajectory` 从 `aux_outputs[:-1]` 读取解码层，排除最后一个 encoder top-k 辅助输出，再追加最终 `pred_boxes` |
| 解码层数 | R18 为 3 层：B0、B1、B2；`last_2` 实际选择 B0→B1、B1→B2，两条全部可用 transition |
| eval 关闭位置 | `det_solver.py` 的逐轮验证和 `val()` 均调用独立 `det_engine.evaluate`；该函数只执行正常 `model(samples)`，没有 MERT plugin、shift、配对或轨迹损失 |

真实参数如下；组合配置通过 YAML 继承原配置，没有第二套 MERT 参数。

| 参数 | 当前 late_xywh 值 |
| --- | --- |
| trajectory weight / beta | 0.10 / 0.01 |
| xy_weight / wh_weight | 1.0 / 0.25 |
| layers / loss_type | last_2 / smooth_l1 |
| shift max_pixels / choices | 1 / [-1, 0, 1] |
| forbid_zero_zero / per_image / fill_mode | true / true / baseline |
| shifted_detection_loss | true：检测 criterion 监督 concat 双视图 |
| small-object weighting | enabled=true；reference_area=256，gamma=0.5，权重范围 [1,4]，max_object_area=null |
| matching | 最后一层独立匹配；排除 DN queries |
| require_fully_visible | true |
| schedule | disabled：200 轮保持 MERT weight=0.10，而非仅后半程开启 |

## 2. CTER 数学与工程实现

主文件：`src/solver/cter_loss.py`。独立接口为：

```python
loss, optional_debug_sums = CTERLoss(config, feature_strides, stage_names)(
    features=original_backbone_features,
    targets=original_targets,
    image_sizes=actual_post_resize_image_sizes,
)
```

`det_engine.train_one_epoch` 仅当 `CTER.enabled=true` 时创建训练插件。
stride 从实际 `backbone.out_strides` 获取，stage 名从 `backbone.return_idx` 获取。
本工程 R18 实际输出 S3/S4/S5，stride 为 8/16/32，CTER 内没有硬编码这个 stride 表。

`RTDETR.forward` 只增加可选 `return_backbone_features=True` 接口：在原 backbone 计算完成后返回已有的 feature 引用和真实输入尺寸。
正常 forward 的返回值仍是原输出字典；PResNet.forward 没有改动，feature 没有增强、替换或改写。
CTER 不安装永久 hook，不挂在 detector 上，没有 Parameter、Conv、Linear、head、memory bank 或跨 batch prototype。

### 2.1 Soft occupancy 与背景环

对于真实 stride 对应的像素 cell C 和 GT 框 B：

```text
w+(cell) = Area(cell ∩ B) / Area(cell)
```

通过 x/y 交叠长度广播相乘计算所有空间 cell，边缘 cell 按裁到真实输入边界后的面积计算。
没有 floor/ceil 硬裁 GT，也没有 Python h/w cell 循环。允许按图像和 GT 实例循环。
GT 使用训练数据原有的归一化 cxcywh，乘以 forward 实际多尺度尺寸，而不是原始照片的 orig_size。

背景是扩大框与全部 GT 并集的几何差集：

```text
context = Expand(B, context_scale) ∩ image
background = context \ Union(all GT boxes)
w-(cell) = Area(cell ∩ background) / Area(cell)
```

实现将 context 内 GT 的 x/y 边界排序、形成矩形分区，用广播判断分区是否位于任一 GT 内，再用矩阵乘法计算 cell 的剩余背景面积。
因此自身 GT、其他 GT、重叠 GT 的并集都被精确排除；不是简单相加掩码而造成重叠区域重复扣除。
同一粗 cell 可包含互不重叠的目标像素和背景像素，所以该 cell 的 w+、w- 可以同时非零；这不是把 GT 像素当背景。

### 2.2 即时 prototype、hard background、margin

所有 CTER 数值计算局部禁用 autocast，使用 FP32。
每个 stage 在 channel 维做 `F.normalize(..., eps=1e-6)`，记归一化特征为 z。
实例 prototype 只由本图、本 GT、本 stage 计算：

```text
p = Normalize(Σ w+ z / (Σ w+ + eps))
q(cell) = dot(p, z(cell))
A+ = Σ w+ q / (Σ w+ + eps)
```

背景使用 soft occupancy 加权 LogMeanExp：

```text
A- = τ [logsumexp(q/τ + log(w-)) - log(Σ w-)]
M = A+ - A-
```

这是对请求中二值背景公式的显式 soft 扩展：当有效背景 cell 的 w-=1 时，严格退化为 `τ log(Σ exp(q/τ) / N-)`。
分数面积用于保留极小目标背景环；不是普通 background mean。
计算使用 `torch.logsumexp`，零权重项置为 -inf；空背景使用有限值和有效性 mask，避免全 -inf 的无效梯度。
默认 τ=0.10，没有直接 exp 求和。

### 2.3 Evidence relay

完整公式：

```text
reference_s = γ * stop_gradient(ReLU(M_s))
L_s→t = ReLU(reference_s - M_t)^2
a_t = Σ w+_t
ω_s→t = Clamp(a_t, 0, 1)
L_CTER = Mean_GT[valid_34 * ω_34 * L_34 + valid_45 * ω_45 * L_45]
L_total = L_det + 0.05 * L_CTER + 0.10 * L_MERT_raw
```

`detach()` 位于 source margin 的 ReLU 之后；source reference 本身不产生梯度。
**这不等于冻结 S3/S4**：destination feature 依赖上游 backbone，因此 destination 路径仍可以更新早期层。
valid 要求该 transition 两个 stage 均有正目标面积及有效背景；无效实例贡献 0，但均值分母仍为全部 GT。
DDP 用全局 GT 数归一化，并补偿 DDP 梯度均值；空 GT rank 也参与同一个分母 collective，避免死锁。
stage weight 只依赖目标在下一阶段的连续 occupancy，不使用 COCO small/medium/large 阈值。
支持 transitions `[3to4]`、`[4to5]`、`[3to4,4to5]`，context_scale 1.25/1.5/2.0，loss_weight 0.02/0.05/0.10；本次默认不扫参数。

### 2.4 两种训练策略互相独立

CTER 接口没有 shifted view、shift metadata、Hungarian、query、decoder 输出或 trajectory 输入。
MERT 文件没有 CTER 的 margin/prototype/loss 依赖。
combo 在 MERT 的 2B concat 输出中切取前 B 张原图的现有 S3/S4/S5，CTER 不使用 shifted feature，也不增加 backbone forward。

| 模式 | 顶层训练 model 调用/step | 实际处理视图 | 推理 model 调用/batch |
| --- | ---: | ---: | ---: |
| Baseline | 1 | B | 1 |
| CTER-only | 1 | B | 1 |
| MERT concat | 1 | 2B | 1 |
| CTER + MERT concat | 1 | 2B | 1 |

MERT 关闭时 trainer 不创建 plugin、不执行 prepare；CTER 关闭时不请求额外 feature 返回、不创建 loss plugin。
eval/test-only 均不创建这两个训练插件，只走原 detector 路径。

### 2.5 Debug

`CTER.debug=true` 时统计全部 S3/S4/S5 的 margin/occupancy 均值，以及启用 transition 的 observed ratio、loss、有效 GT 数。
ratio 只统计正 source margin 的有效实例；debug 的 loss_34/loss_45 是 validity 加权后的未乘 0.05 均值。
batch 内只累积 detached GPU 标量，epoch 末打包一次 all_reduce、一次 CPU 拷贝。
空 GT rank 的 debug key 集合相同。
`debug=false` 不执行这些统计的 `.item/.cpu/.numpy/.tolist/synchronize`；全局 loss 分母归一化和原训练日志仍有各自必需的通信。

## 3. AMP 修复、测试入口与测速

原问题在 `src/solver/det_engine.py:evaluate`：评估 model forward 外的 autocast 被注释，且逐轮验证/test-only 没有传入 AMP 开关。
现在训练、逐轮 validation、test-only、`tools/infer.py` 的整图/切片推理、benchmark、既有 micro-shift 诊断工具都调用 `src/misc/amp.py:autocast_context`。
CUDA `--amp` 使用 FP16 autocast；CTER 保留 FP32 损失计算。GradScaler 仅在 `BaseSolver.train()` 创建，eval 不读取 cfg.scaler。
每轮训练打印 Training AMP；每次 validation/test-only 开始打印一次 `Evaluation AMP: enabled/disabled`。

`--debug-eval-amp` 在第一个真实评估 batch 安装临时 hook，打印 input、S3/S4/S5、encoder、decoder logits/boxes dtype，实际执行的 Conv/Linear/Attention 输出 dtype、实际 autocast 状态。
同时记录 model/backbone/encoder/decoder 的真实调用次数；任何一项不等于 1 均报错。AMP debug 还检查第一实际 Conv 输出为 FP16。
hook 在第一 batch 后移除，测速时审计在计时区间之外进行。
input 仍可能 FP32，部分 residual/数值敏感算子和最终 boxes 保持 FP32 是正常的，不强制所有张量 FP16。
**具体 dtype 值尚未在本机实测，不能预填。**
既有 `analyze_micro_shift_equivariance.py` 是显式多平移诊断，保留其主动执行多个 shifted forward 的用途，不把它当普通 test-only 或纯推理测速；新增 --amp 也用同一个 helper。
`export_onnx.py` 仍按原来的 CPU/FP32 图导出，没有新增训练分支或强制混合精度导出。

test-only `--test-num-workers` 默认 8，支持显式设为 0 或其他非负值，并打印旧值→新值；它不会改变训练期间 validation 的 YAML worker 数。
现有 val_dataloader 默认指向 **val split**；正式 test split 的跑法见下方，不能把 --test-only 自动等同于 DUT test split。

`tools/benchmark_inference.py` 使用预生成 GPU 输入、CUDA Events，默认 batch=1、warmup=100、iters=500。
只测未 deploy 的正常 detector forward；没有 DataLoader、磁盘读图、COCO evaluator 或 postprocess。
先严格加载 checkpoint（优先 EMA.module），再 warmup，计时结束一次同步读取 Events。
打印配置、checkpoint、input size、batch、AMP、Params、mean/median/P95 latency、FPS、峰值 allocated 显存。
`--compare-configs` 可用同一个 checkpoint 自动比较四种 YAML 并输出表格；`--output` 可保存 JSON。
测速输入必须与 encoder/decoder 配置的 eval_spatial_size 一致，当前默认 800×800。

| Variant | Params | 推理 FLOPs | Forward latency / FPS |
| --- | --- | --- | --- |
| Baseline | 待服务器实测绝对值 | 原检测网络 | 待 CUDA Events 实测 |
| MERT | 与 Baseline 相同，新增 0 | 与 Baseline 相同 | 待 CUDA Events 实测 |
| CTER | 与 Baseline 相同，新增 0 | 与 Baseline 相同 | 待 CUDA Events 实测 |
| CTER+MERT | 与 Baseline 相同，新增 0 | 与 Baseline 相同 | 待 CUDA Events 实测 |

参数增量为 0、推理张量算子/形状不变是代码结构结论，不是假定的测速结果；理论 FLOPs 相同，绝对 FLOPs 没有另行测算。
相同硬件、AMP、输入和 checkpoint 下 latency 应接近，但不承诺逐次完全相等；避免与其他 GPU 任务并行测速，重复测几次。
新评估 AMP 的数值结果与旧 FP32 测试可能略有差别，回归/公平对比必须统一 AMP 设置。

## 4. 配置与训练公平性

下表文件均位于 `configs/rtdetr/`；旧文件保留，新增 6 个实验文件及公共 `include/cter.yml`。

| YAML | 实验 |
| --- | --- |
| `rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp0_baseline.yml` | 现有公平 Baseline，两个模块均关闭 |
| `rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh.yml` | 现有已验证 MERT，CTER 默认关闭 |
| `rtdetr_r18vd_200e_dut_anti_uav_cter_34.yml` | CTER S3→S4 only |
| `rtdetr_r18vd_200e_dut_anti_uav_cter_45.yml` | CTER S4→S5 only |
| `rtdetr_r18vd_200e_dut_anti_uav_cter_345.yml` | CTER 两条 relay only |
| `rtdetr_r18vd_200e_dut_anti_uav_cter_mert_late_xywh.yml` | CTER-345 + 原 MERT |
| `rtdetr_r18vd_200e_dut_anti_uav_cter_34_mert_late_xywh.yml` | CTER-34 + 原 MERT |
| `rtdetr_r18vd_200e_dut_anti_uav_cter_45_mert_late_xywh.yml` | CTER-45 + 原 MERT |

已解析全部 YAML 检查结构与训练策略一致，组合 MERT 字典与原 late_xywh 字典完全相等。
统一 200 epochs、每卡原图 batch=10（3 卡原图 batch=30，MERT forward 视图 batch=60）、train workers=8/卡、val workers=0/卡。
固定验证 800×800、训练原多尺度最高 960；AdamW 全局 lr=1e-4、backbone lr=1e-5，weight decay=1e-4（原 norm/bias 规则保留），milestones=[120,170]、gamma=0.1。
augmentation、dataset、criterion、初始化、seed、evaluation 均沿用原配置；PResNet pretrained=true（既有 backbone 初始化），不自动加载 COCO 检测 checkpoint。
**不要拿 batch=24 的 `rtdetr_r18vd_6x_dut_anti_uav.yml` 直接与这些 batch=10 实验做公平对比。**
另修复 `load_config` 的可变默认字典，防止同一进程先加载 CTER 配置再加载原配置时开关泄漏。

## 5. 验证结果与测试覆盖

本机无 PyTorch，没有安装新依赖，也没有启动训练。

- 已通过：新增/修改 Python 文件编译检查、YAML 解析、8 个配置训练策略一致性、组合 MERT 参数完全一致、独立 config 加载无开关泄漏、git diff whitespace 检查。
- 已通过源码差异检查：`mert.py`、原 late_xywh YAML、PResNet 无修改；src/configs/tests/tools 无旧实验模块依赖。
- 已尝试：`python -m unittest tests.test_cter tests.test_mert -v`，由于 `ModuleNotFoundError: No module named 'torch'` 在导入阶段未执行测试；不是单元测试已通过。
- 尚待服务器：数学单元测试、四模式实际 forward 数、原 MERT/原 Baseline 单步及推理回归、CUDA dtype、参数绝对值、latency/FPS、完整训练和 AP/AR。

新增 `tests/test_cter.py` 覆盖：完整/半个/subcell/border occupancy，自身及邻近重叠 GT 背景排除，分数面积环，相同/相反特征 margin，hard-negative LSE，relay 满足/违反、source detach、空 GT/背景、debug=false 无统计 CPU 同步、空 rank 分母 collective、debug key 对齐、四模式训练调用与 shift 次数、CTER 关闭后的原检测/MERT 单步一致性、CTER 无直接 encoder/decoder 梯度、真实 R18 四配置参数/key/同权重 eval 输出一致、eval 无插件初始化、eval 不访问 GradScaler、可选 CUDA 实际 dtype。
原 `tests/test_mert.py` 保留不改。

服务器同步文件后先运行：

```bash
python -m unittest tests.test_cter tests.test_mert -v
```

## 6. 运行命令

在服务器 `RT-DETR-UAV/rtdetr_pytorch` 目录运行。按顺序运行，不要在同三张卡上并发这些训练。
每次新的实验不加 -r，不覆盖原输出目录；已有同名 output/log 时先另存，避免混合实验记录。

### Exp0：原 MERT 回归

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nproc_per_node=3 --master_port=9911 tools/train.py \
-c configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh.yml \
--amp --seed 0
```

### Exp1：CTER-345 only

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nproc_per_node=3 --master_port=9911 tools/train.py \
-c configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_cter_345.yml \
--amp --seed 0
```

### Exp2：CTER-345 + MERT（核心实验）

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nproc_per_node=3 --master_port=9911 tools/train.py \
-c configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_cter_mert_late_xywh.yml \
--amp --seed 0
```

Baseline、34、45 或 34/45+MERT 只替换 -c 为表格对应 YAML；其余参数一致。

### Validation/test-only 与真实 AMP 检查

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
-c configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_cter_mert_late_xywh.yml \
-r output/rtdetr_r18vd_200e_dut_anti_uav_cter_mert_late_xywh/best.pth \
--test-only --amp --debug-eval-amp --test-num-workers 8 --seed 0
```

以上使用 YAML 的 val_dataloader，默认是 **val split**，不是自动切换到 DUT test split。
若要评估独立 test split，请复制对应实验 YAML 用于评估，继承该实验并仅覆盖数据路径，例如：

```yaml
__include__: ['./rtdetr_r18vd_200e_dut_anti_uav_cter_mert_late_xywh.yml']
val_dataloader:
  dataset:
    img_folder: ../DUT-Anti-UAV/DUT-Anti-UAV/images/test/
    ann_file: ../DUT-Anti-UAV/DUT-Anti-UAV/labels/test.json
```

保持 --test-only 命令及原实验的 checkpoint 不变，-c 指向这份评估 YAML。不要用 test split 参与选最佳模型。
首次开始会打印 `Evaluation AMP: enabled`；debug 输出的各组件 forward count 应实际全为 1。
普通评估建议去掉 --debug-eval-amp，避免第一 batch 诊断 hook 的额外耗时。

### 单模型纯 forward Benchmark

```bash
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_inference.py \
-c configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh.yml \
-r output/rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh/best.pth \
--amp --batch-size 1 --warmup 100 --iters 500 --debug-eval-amp
```

### 同 checkpoint 验证四种配置零推理结构增量

```bash
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_inference.py \
-c configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp0_baseline.yml \
--compare-configs \
configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh.yml \
configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_cter_345.yml \
configs/rtdetr/rtdetr_r18vd_200e_dut_anti_uav_cter_mert_late_xywh.yml \
-r output/rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh/best.pth \
--amp --batch-size 1 --warmup 100 --iters 500 --debug-eval-amp \
--output benchmark_cter_amp.json
```

不需要先训练 CTER 就可做此结构对比，因为四种配置 checkpoint key 完全相同。
比较 FP32/AMP 时复制命令去掉 --amp，使用另一个 --output，保持 GPU 无其他任务并重复测速。

## 7. 风险与实验判读

- 极小 GT 与背景可能落入同一粗 cell，特征无法在 cell 内区分；soft occupancy 防止几何支持消失，但不会创造新的高分辨率特征。stage weight 会减弱不可靠监督，不能保证小目标 recall 增长。
- source margin 接近 0 时可供 relay 的证据有限；CTER 没有独立的绝对分离目标，不保证自身学到强正 margin。
- detach 只切断 reference 梯度，destination loss 仍会通过共享上游 backbone 反传；不要声称彻底阻止所有早期 margin 变化。
- 名义 stride cell 是监督几何模型，不等于实际卷积感受野的精确范围。
- 精确排除 GT 并集的矩形分区随 GT 数增加开销增大；UAV 每图少量 GT 时适用，拥挤多目标场景需重新 profiling。
- CTER 有额外 FP32 feature/mask 计算、反传与 GT 分母通信，可能增加训练时间/显存；零推理结构开销不等于零训练开销。
- λ=0.05、context=1.5、τ=0.10、γ=0.9 是首轮设计选择，不是已经验证最优参数。
- 比较 AP50:95、AP50、AP75、AP-small/medium/large、AR@100、AR-small/medium/large，尤其看 recall 是否改善；不要只比较某一 epoch 的最高 AP。
- MERT≈63.9 是既有实验结果，CTER+MERT 达到 64.3+ 仅是研究期待，不作承诺；组合不增益也不能仅凭 AP 数值断言目标完全重叠，需看统计、recall 和重复实验。
- 先原 MERT 回归→CTER-345→345+MERT；若组合不理想再做34+MERT、45+MERT，不同时修改 MERT 或扫大量权重。论文结论建议使用多个 seed 与一致数据/训练/AMP 协议。

## 8. 修改文件清单

- 新增：`src/solver/cter_loss.py`、`src/misc/amp.py`、`src/misc/inference_audit.py`、`tools/benchmark_inference.py`、`tests/test_cter.py`、公共 CTER YAML、6 个实验 YAML、本报告。
- 修改：`src/zoo/rtdetr/rtdetr.py`（可选返回已有 feature）、`src/solver/det_engine.py`（独立 loss 接线/eval AMP）、`src/solver/det_solver.py`（传递开关）、`src/solver/solver.py`（scaler 只在 train 创建）、`src/core/yaml_utils.py`（隔离配置加载）、`tools/train.py`（dtype debug/test workers）、`tools/infer.py`（统一整图/切片推理 AMP）、`tools/analyze_micro_shift_equivariance.py`（诊断 forward 的 AMP 开关）。
- 保留不改：原 MERT 源文件及 late_xywh YAML、现有公平 Baseline YAML、PResNet/HybridEncoder/Decoder/Matcher/criterion 实现。
