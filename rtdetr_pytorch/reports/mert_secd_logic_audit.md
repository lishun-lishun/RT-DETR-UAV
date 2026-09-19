# MERT / SECD 实现逻辑审计

审计对象：`E:/code/RT-DETR/RT-DETR/rtdetr_pytorch`

结论：MERT 的图像/GT 平移、坐标方向、边界裁剪、稳定 GT 身份、最终层独立 Hungarian、固定 query 轨迹、late-xywh、像素面积权重及训练/测试隔离均正确；SECD 的公式、梯度路径和两个真实 stride transition 均正确。发现并最小修复 1 个生产 BUG：MERT 在 DDP 下按每个 rank 的本地有效 pair 数独立归一化，而不是按全局有效 pair 数归一化。

## 1. 当前代码地图

| 项目 | 文件、符号与关键位置 |
| --- | --- |
| MERT 核心 | `src/solver/mert.py`，`MERT`（260）、`MERT.prepare`（306）、`MERT.calculate_loss`（345） |
| MERT loss | `src/solver/mert.py:345-406`，检测 loss 组合在 `src/solver/det_engine.py:31-44` |
| shifted image | `src/solver/mert.py:71-86`，`shift_images` |
| shifted GT | `src/solver/mert.py:113-194`，`shift_targets` |
| Hungarian | 原 matcher 为 `src/zoo/rtdetr/matcher.py:20-108`；MERT 最终层两视图调用为 `src/solver/mert.py:365-371` |
| Decoder trajectory | `src/solver/mert.py:197-207`，`decoder_box_trajectory`；源 decoder box 在 `src/zoo/rtdetr/rtdetr_decoder.py:260-278` |
| MERT 配置 | `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_mert_late_xywh.yml:7-24` |
| SECD 核心 | `src/nn/backbone/secd.py:12-83`，`SECDTransition` |
| SECD 挂载 | 构造在 `src/nn/backbone/presnet.py:193-217`；执行/相加在 `src/nn/backbone/presnet.py:308-317` |
| SECD-34 | `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_34.yml:7-15` |
| SECD-45 | `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_45.yml:7-8`，其余参数继承 SECD-34 |
| SECD-345 | `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_345.yml:7-8`，其余参数继承 SECD-34 |
| SECD34+MERT | `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_34_mert_late_xywh.yml:1-11` |
| SECD345+MERT | `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_345_mert_late_xywh.yml:1-8` |
| 训练入口 | `src/solver/det_solver.py:37-43` 将 MERT 配置传入；`src/solver/det_engine.py:60-100` 仅训练时构造/执行 |
| Baseline decoder 数 | `configs/rtdetr/rtdetr_r18vd_6x_coco.yml:25-28`，R18 为 3 层 |

## 2. PResNet 真实尺度（实际运行 1×3×640×640）

| 目标项目名称 | shape | stride | 是否交给 HybridEncoder |
| --- | --- | ---: | --- |
| Stem + max-pool | `[1,64,160,160]` | 4 | 否 |
| S2 / `res_layers[0]` | `[1,64,160,160]` | 4 | 否 |
| S3 / `res_layers[1]` | `[1,128,80,80]` | 8 | 是 |
| S4 / `res_layers[2]` | `[1,256,40,40]` | 16 | 是 |
| S5 / `res_layers[3]` | `[1,512,20,20]` | 32 | 是 |

静态定义见 `src/nn/backbone/presnet.py:175-191`。真正的 stride=2 发生在 S3、S4、S5 各自第一个 BasicBlock；S2 第一个 block 的 stride 为 1（`src/nn/backbone/presnet.py:120-132`）。

因此：

- `SECD-34` 实际是 **stride 8 → stride 16**（128 → 256 channels）。
- `SECD-45` 实际是 **stride 16 → stride 32**（256 → 512 channels）。
- 名称与真实尺度一致，没有 stage off-by-one。

## 3. MERT 逐项审计

### 3.1 shift 采样与发生位置

- `MERT.prepare` 在 DataLoader augmentation、640 resize、归一化 `cxcywh` 完成以后，进入 detector forward 以前执行（`det_engine.py:67-75`）。
- 若 RT-DETR 多尺度开启，它先使用与原 `RTDETR.forward` 相同的 NumPy RNG 采样一次尺寸并插值（`mert.py:312-316`），随后临时关闭模型内部第二次 resize（`mert.py:49-68`）。所以 1 pixel 总是针对本次真正送入模型的 `H×W`，不是原始图尺寸或 feature 尺寸。
- 随机候选由 `{-1,0,1}² \ {(0,0)}` 的 8 个方向构成（`mert.py:319-329`），与目标定义一致。

### 3.2 图像方向与 wrap-around

- 当前没有使用 `torch.roll`、`grid_sample` 或 affine grid。
- `shift_images` 创建全零输出并执行源/目标切片复制（`mert.py:71-86`），暴露边缘为 0，不存在 wrap-around。
- 数值测试：100×100 图像的亮点 `(x=50,y=50)` 在 `dx=+1,dy=0` 后位于 `(51,50)`。所以正 `dx` 定义为图像内容向右，正 `dy` 为内容向下。

### 3.3 GT 方向、坐标空间与 inverse shift

- 训练 transform 最终把 target 转成 normalized `cxcywh`（`src/data/transforms.py:144-168`）。
- `shift_targets` 把 `[dx,dy,dx,dy]` 除以 `[W,H,W,H]` 后加到 normalized `xyxy`（`mert.py:162-169`）。所以 `dx=+1` 精确产生 `cx += 1/W`，x/y、符号和尺寸来源都正确。
- `inverse_box_shift` 对预测中心执行 `cx -= dx/W, cy -= dy/H`，宽高不动且不裁剪预测（`mert.py:89-95`），符号正确。
- Decoder 每层 box 都由 sigmoid 得到（`rtdetr_decoder.py:260-267`）。实际运行得到 3 个 `[1,300,4]` decoder 输出，值均在 `[0,1]`，不存在 logit box 与 sigmoid box 混减。

### 3.4 GT clipping 与身份

- shift 前先转 normalized `xyxy`，shift 后 clamp 到 `[0,1]`，再删除宽/高为 0 的完全移出目标（`mert.py:155-177`）。不会产生负宽高或中心越界。
- 部分移出目标保留 clipped box 参加 shifted detection loss，但 `fully_visible=False`，不会参加 MERT trajectory loss；完全移出目标从 shifted detection GT 中删除。该策略明确且一致。
- `origin_gt_id` 在当前 augmentation 后的 target 上创建或验证唯一性（`mert.py:151-160`），shift/filter 后随 instance fields 一起索引（`mert.py:171-177`）。它只需在同一图像的 original/shifted view 间稳定，不依赖 GT 数组位置。

### 3.5 Hungarian 与 trajectory

- Original 和 Shifted 的最终输出分别调用一次原 `HungarianMatcher`（`mert.py:365-371`）。
- 两边通过 `origin_gt_id` 连接同一 GT，即使 GT 顺序或 query index 不同也正确（`mert.py:375-389`）。
- 中间层不重新匹配。最终匹配得到的 original/shifted query index 被固定，再沿 decoder trajectory 读取（`mert.py:390-394`）。不存在将不同 query 拼成轨迹的问题。
- `decoder_box_trajectory` 明确排除 DN 输出和最后一个 encoder proposal aux，只收集 decoder aux + final（`mert.py:197-207`）。R18 实际为 B0、B1、B2，共两段 refinement。
- `last_2` 在实现中含义是最后 **2 个 transition**，因此取最后 3 个 box（`mert.py:361`、`390-394`），没有 off-by-one。

### 3.6 late_xywh 数学与 reduction

实现确实计算：

`Δb(l)=b(l)-b(l-1)`，然后比较 original delta 与 inverse-aligned shifted delta（`mert.py:390-395`），不是最终框 consistency。

`smooth_l1_loss(..., beta=self.beta, reduction='none')` 真实传入 beta（`mert.py:395-396`）；坐标权重为 `(xy_weight,xy_weight,wh_weight,wh_weight)`（`mert.py:372-373`）。当前 YAML 对应：

`Lxywh = Lcx + Lcy + 0.25(Lw+Lh)`，`beta=0.01`。

单卡 reduction 是所有有效 matched/visible GT pair 和所选 transition 的加权和除以 `N×K`。无 pair 时返回由两视图预测构造的可微、同 device、FP32 零值。

### 3.7 small-object weighting

box 宽高确实是 normalized，但代码先乘本次实际 `width×height`：

`area_pixel = w_norm × h_norm × W × H`（`mert.py:337-341`）。

然后才计算 `clamp((256/(area_pixel+eps))^0.5,1,4)`。不存在把 normalized area 直接和 256 比较的尺度错误。

### 3.8 shifted detection loss

当前正式配置是 `forward_mode=concat` 且 `shifted_detection_loss=true`。一次 forward 得到 2B batch，并把 original+shifted target 一起交给未经修改的 criterion（`det_engine.py:22-37`）。因此实际总损失为：

`L = Ldet_joint(original, shifted) + 0.10 × Ltrajectory`

当两视图 GT 数量一致时，criterion 的目标数归一化使 `Ldet_joint` 等价于 `0.5×Ldet_ori + 0.5×Ldet_shift`，不是把完整 detection loss 无意翻倍。若边缘 GT 在 shifted view 完全移除，则它按联合有效 GT 总数归一化，不再严格等于两个独立均值的 0.5/0.5。

Sequential 模式单卡路径显式调用 `average_loss_dicts`；多卡禁止 sequential，避免两个 DDP forward 带来的同步错误（`det_engine.py:38-43,63-65`）。

### 3.9 train/eval 路径

- MERT 仅在 `train_one_epoch` 且 YAML enabled 时构造；`prepare` 对 eval model 直接返回 None。
- 验证/测试没有 shifted image、第二次 forward 或 MERT loss。
- MERT 是普通 solver 对象，不进入 detector state_dict，不改变推理结构、参数量或速度。

## 4. 明确发现并修复的 BUG

### 当前实现（修复前）

每个 DDP rank 使用自己的 `pair_count_r`：

`L_r = λ T_r / (N_r K)`

DDP 再平均各 rank 梯度。这样每张卡权重相同，而不是每个有效 GT pair 权重相同。若某卡 1 pair、另一卡 10 pairs，两张卡仍各占 50%；若某卡 0 pair，它返回零并仍稀释其他卡的 MERT 梯度。

### 为什么错误

所需目标是所有 GPU 上有效 pair 的全局平均。有效 pair 数会因每张图 GT 数、Hungarian 结果和边界 `fully_visible` 过滤不同而变化，所以不能假设每卡相同。

### 正确数学含义

令 `N̄=(Σ_r N_r)/world_size`。每个 rank 返回：

`L_r = λ T_r / (N̄ K)`

DDP 的梯度平均后恰好得到：

`λ Σ_r T_r / ((Σ_r N_r)K)`。

这与原 RT-DETR criterion 使用“全局目标数 / world_size”的归一化约定一致。

### 影响

- 多卡 MERT 对各 rank 数据分布敏感，loss 强度不稳定。
- 边界目标/空目标分布不均时偏差最大。
- 单卡不受该 BUG 影响，因此过去单卡单元测试无法发现。

### 最小修改

只修改 `src/solver/mert.py:351-406`：所有 DDP rank 都参与一次 `all_reduce(pair_count)`，以全局 mean pair count 为分母；本地 0 pair 的 rank 保留连接计算图的零 loss。没有修改 MERT 公式、lambda、matcher、检测 loss 或配置。

## 5. SECD 审计

- 四个 phase 的顺序和 shape 正确（`secd.py:64-67`）。奇数高宽仅在右/下 replicate pad，使结果为 ceil(H/2)×ceil(W/2)，与 PResNet stride-2 分支一致。
- residual、group energy、temperature softmax、kappa、加权 residual 和 reshape 均逐项符合定义（`secd.py:68-78`）。低精度能量/softmax 局部提升 FP32，输出转回输入 dtype。
- projection 是 1×1 `ConvNormLayer`，仅处理已 2× 下采样 evidence（`secd.py:42,80-83`）。
- 最终是 `F_base + alpha_eff×evidence`；`alpha_eff=0.2*tanh(raw_alpha)`，初值 raw=0 时输出严格等于 Baseline。
- 分支在 alpha=0 时仍执行，因此 raw_alpha 有梯度；projection 参数第一步为零梯度，raw_alpha 更新后即可获得梯度。这是零门控 residual adapter 的预期行为，不是断梯度。
- SECD-34、45 的输入分别是真实 S3 /8、S4 /16，输出分别与完整原 S4 /16、S5 /32 相加（`presnet.py:308-317`），未替换任何原 stage。
- SECD-345 是级联结构：增强后的 S4 同时进入原 S5 和 45 bypass。这是组合配置的自然语义，不是两个完全独立于上游增强的旁路。
- train/eval 都启用 SECD，这是结构模块的正确行为；预训练仅缺少新增 `secd_*` key，原骨干 key 保持兼容。

未发现 SECD 公式、stage、shape、AMP、梯度或推理路径 BUG，因此没有修改 SECD 生产代码或参数。

## 6. 配置和禁止项核对

解析后的 Baseline、MERT、SECD34、SECD45、SECD345、SECD34+MERT、SECD345+MERT 审计通过。各 MERT/SECD 配置除开关/模块参数和 output_dir 外，继续继承同一份：输入、多尺度、augmentation、batch、optimizer、LR、scheduler、200 epoch、HybridEncoder、decoder、matcher 和 detection loss。

本次没有修改任何禁止项，也没有调 MERT/SECD 超参数。

## 7. 测试证据与限制

- 实际 640 输入 backbone hook：S2 `[1,64,160,160]`、S3 `[1,128,80,80]`、S4 `[1,256,40,40]`、S5 `[1,512,20,20]`。
- 实际 R18 train forward：3 个 decoder box outputs，均为 `[1,300,4]` normalized sigmoid box；另有 3 个 DN aux，trajectory 未混入 DN/encoder。
- MERT：25 个适用的 CPU 数值/真实 decoder/matcher/criterion 测试通过，包括 8 方向无 wrap、GT shift/inverse、边界裁剪、身份重排、固定 query trajectory、late-two、beta、xywh 权重、像素面积、空 pair、DN split、backward 及新增 DDP normalizer mock regression。
- 本地 torchvision 0.25 已删除工程所用的旧 `torchvision.datapoints` API，因此 1 个生产 metadata transform 测试无法在本机导入；服务器原训练环境不受本机版本不匹配结论替代，仍应运行完整测试。
- SECD：23 个测试通过，包括公式独立参考、phase permutation、奇数 shape、FP16/BF16、有限梯度、零门控等价、预训练 key、freeze 和各 transition。
- 新增独立 Linux/Gloo 两 rank regression：`tests/test_mert_ddp_reduction.py`。当前 Windows PyTorch 构建不支持本机 Gloo device，所以本机按平台跳过；应在 Linux 训练服务器运行。
- 现有真实 R18+SECD+MERT 两 rank 测试 `tests/test_mert_distributed.py` 已增加不等 pair 数回归，但因上述本地依赖版本不能在当前 Windows 环境完整执行。

服务器建议验证命令：

```bash
python -m unittest tests.test_mert tests.test_secd tests.test_mert_ddp_reduction tests.test_mert_distributed -v
```

该命令只运行 synthetic/debug tests，不启动正式训练。
