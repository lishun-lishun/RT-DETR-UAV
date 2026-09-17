# 原始 RT-DETR R18 增量开发：DUT / MERT / SECD 完整变更报告

本次以修改前的当前干净工作树为准，不恢复旧实验工程。先完成原版审计，再新增数据适配、训练期 MERT 和 Backbone 旁路 SECD。没有执行正式训练、完整 DUT 评估或服务器 CUDA/NCCL 测试。

## 1. 原版审计与统一协议

原版入口为 `tools/train.py`，R18 配置为 `configs/rtdetr/rtdetr_r18vd_6x_coco.yml`。该 YAML 继承原版 COCO dataset、runtime、dataloader、optimizer 和 `rtdetr_r50vd.yml` 模型定义，再覆盖 R18 的 Backbone/Encoder/Decoder 设置。DUT 配置直接继承它，而不是另建一套训练超参数。

| 项目 | 当前原版实际值；七种方法全部继承 |
| --- | --- |
| Backbone | PResNet18-vd，variant=d，4 stages，return_idx=[1,2,3]，freeze_at=-1，freeze_norm=False，pretrained=True |
| S3/S4/S5 | channels=[128,256,512]；strides=[8,16,32] |
| HybridEncoder | 输入 S3/S4/S5，hidden_dim=256，expansion=0.5，encoder layers=1，nhead=8 |
| Decoder | 3 decoder layers，300 queries，100 denoising queries，eval_idx=-1 |
| 训练轮数 | 72；未增加到 200，文件名不使用误导性的 200e |
| Train base resize | 640×640 |
| Train random multi-scale | [480,512,544,576,608,640,640,640,672,704,736,768,800]，直接继承原版列表 |
| Validation / Test | 640×640 |
| eval_spatial_size | Encoder 和 Decoder 均为 [640,640] |
| Batch size | train=4/GPU，val/test=8/GPU；3 卡训练原图全局 batch=12 |
| Worker | train/val/test=4/进程；未引入旧版 worker=8、预取或 pinned-memory 补丁 |
| Optimizer | AdamW，betas=[0.9,0.999]，weight_decay=1e-4；原有 norm/bias 免 decay 分组不变 |
| LR | Backbone=1e-5，其余=1e-4；SECD 属于 Backbone，沿用原分组，不另调 LR |
| Scheduler | MultiStepLR，milestones=[1000]，gamma=0.1；原版 72 轮内不触发该衰减，没有另加 scheduler warmup |
| Gradient clipping | max_norm=0.1 |
| EMA | enabled=True，decay=0.9999，warmups=2000；这是 EMA warmup，不是 LR warmup |
| AMP | YAML 默认 False；使用原版 --amp 开启训练 AMP；验证/测试仍为原版 FP32 |
| DDP | sync_bn=True，find_unused_parameters=True；保留原版继承后的实际值 |
| Pretrained | 原版 torch.hub 下载/缓存 `ResNet18_vd_pretrained_from_paddle.pth` Backbone 权重；不是 COCO detector 权重 |

原版训练增强顺序完整保留：RandomPhotometricDistort(p=0.5) → RandomZoomOut(fill=0，库默认最大倍率 4) → RandomIoUCrop(p=0.8) → SanitizeBoundingBox(min_size=1) → RandomHorizontalFlip → Resize([640,640]) → ToImageTensor → ConvertDtype → SanitizeBoundingBox(min_size=1) → ConvertBox(cxcywh, normalize=True)。验证/测试保留 Resize([640,640]) → ToImageTensor → ConvertDtype。

原版 matcher 位于 `src/zoo/rtdetr/matcher.py`，criterion 位于 `src/zoo/rtdetr/rtdetr_criterion.py`，二者未修改。原版 `aux_outputs` 最后一项是 encoder proposal，前两项才是 decoder 中间输出；最终 decoder 输出在 `pred_boxes`。因此 R18 的真实 decoder 轨迹为 B0→B1→B2，有两个 refinement transitions，不将 encoder proposal 或 DN 输出混进去。

原版训练路径：train.py → YAMLConfig → DetSolver.fit → train_one_epoch → RTDETR → PResNet → HybridEncoder → RTDETRTransformer → 原 criterion。每轮验证：DetSolver.fit → evaluate(EMA model) → 单次 detector forward。原版 `--test-only` 调用 DetSolver.val，读取 **val_dataloader**，不是 DUT 的独立 test split。

配置系统仍使用原来的 `@register`、GLOBAL_CONFIG、`__share__`、`__inject__` 和递归 `__include__`。仅为实验隔离在 YAMLConfig 中使用独立解析字典，并给没有声明 SECD 的配置设置默认关闭，避免同进程先加载 SECD 后加载原版配置时开关泄漏。

## 2. DUT 数据集实读结果

数据集根目录相对 `rtdetr_pytorch` 为 `../DUT-Anti-UAV/DUT-Anti-UAV`。直接复用原版 `CocoDetection`，没有新建 Dataset 类、转换标注或修改 COCO loader。

| Split | Image folder | Annotation | 图片数 | 标注数 |
| --- | --- | --- | ---: | ---: |
| train | images/train/ | labels/train.json | 5200 | 5246 |
| val | images/val/ | labels/val.json | 2600 | 2620 |
| test | images/test/ | labels/test.json | 2200 | 2245 |

三个 JSON 的 categories 均为 `[{"id":0,"name":"UAV","supercategory":"object"}]`，所有 annotation.category_id 均为 0；所以 DUT 使用 `num_classes: 1`、`remap_mscoco_category: False`。已检查全部标注引用的图片，缺失数为 0。

训练/验证使用真实 train/val 分割。独立 test 路径放在 `test_dataset` 元数据中，只供新增 `tools/test_dut.py` 选择；训练过程中不读取 test，也不根据 test 选择模型。

## 3. 实际文件变更及必要性

只修改了以下四个已有 Python 文件：

| 文件 | 最小必要修改 |
| --- | --- |
| src/core/yaml_config.py | `load_config(cfg_path,{})` 隔离可变默认字典；默认关闭 SECD，避免配置/全局共享开关跨实验污染 |
| src/nn/backbone/presnet.py | 注入 YAML SECD 设置、独立旁路属性和启用后的 stage-output 残差相加；原 stage 结构及 key 完整保留；预训练允许缺少新增旁路参数，拒绝原参数不匹配 |
| src/solver/det_engine.py | 仅在 train_one_epoch 增加可关闭 MERT 分支；concat 原图/平移图完成单次 DDP forward，调用原 criterion，再加轨迹损失；evaluate 函数未修改 |
| src/solver/det_solver.py | 只将 MERT YAML 设置传给 train_one_epoch，不修改验证、scheduler、checkpoint、EMA 等逻辑 |

新增生产 Python 文件：`src/nn/backbone/secd.py`、`src/solver/mert.py`、`tools/test_dut.py`、`tools/analyze_dut_models.py`。新增测试位于 `tests/`，统计/审计结果位于 `reports/`。

`tools/train.py`、BaseSolver、Encoder、Decoder、matcher、criterion、EMA、dist、DataLoader 和 Dataset 实现保持原样。全部 16 份原有官方 YAML 的原始字节 SHA256 与修改前一致；没有覆盖官方 COCO YAML 或公共 include。

测试辅助文件中的可选依赖 stub 只用于本地无 pycocotools/transformers 环境的单元测试，不进入生产路径。实际模型复杂度统计使用真实模型源码/张量运算，没有模型或依赖 stub；显式 model-only import 只绕过与模型统计无关的 eager import，不代表完整服务器训练依赖验收。

## 4. MERT 的精确定义

MERT 是 solver 侧普通训练插件，不是 nn.Module，不注册 detector 参数、buffer 或推理 hook。`enabled=False` 不构造插件、不调用 prepare、不采样平移、不复制 GT、不增加 detector forward。

开启时先用原版 numpy RNG 从原版 multi_scale 列表采样一次尺寸，执行原版 nearest resize，再在该实际输入尺度生成每张图的非零整数平移 `(dx,dy)∈{-1,0,1}²\{(0,0)}`。像素拷贝，暴露边界填 0，不使用会环绕的 torch.roll。两视图 forward 时临时关闭 detector 内部第二次随机 resize，finally 恢复；多尺度分布未改变。

normalized cxcywh GT 同步平移，原图 GT 保持完整监督。两视图保留稳定的 per-image origin_gt_id。平移后完全出界的 GT 从 shifted detection target 删除；部分裁剪 GT 保留裁剪后的检测监督，但不参与 MERT。正常原图/平移图分别用原版 final-layer Hungarian matcher，依据 origin_gt_id 配对，不假设 query index 相同；未匹配或不可完整观察的 GT 不参与一致性。

shifted 预测框的 cx/cy 分别减去 dx/W、dy/H，w/h 不变；不裁剪预测框。轨迹只取真实 decoder 输出，从最后两次 refinement transition 比较两视图的 box update。

本次坐标求和、GT 与 transition 平均的公式明确为：

```text
Δb_k = b_k - b_(k-1)
Δb_shift_k = inverse(b_shift_k) - inverse(b_shift_(k-1))
A_g = normalized_GT_w * normalized_GT_h * sampled_W * sampled_H
w_g = clamp((256 / (A_g + 1e-6))^0.5, 1, 4)
L_MERT = 0.10 / (N*K) * Σ_(g,k) w_g *
         [SmoothL1(Δcx,Δcx_shift) + SmoothL1(Δcy,Δcy_shift)
          + 0.25*(SmoothL1(Δw,Δw_shift) + SmoothL1(Δh,Δh_shift))]
SmoothL1 beta = 0.01; K = min(2, 实际 decoder transitions)
```

不存在额外的坐标 `/4` 或未经说明的 `/2`，小目标权重不改变分母；无匹配目标/无 transition 时返回可微零损失。损失权重固定，不增加 schedule。

正式 YAML 均使用 `forward_mode: concat`、`shifted_detection_loss: true`。每进程原图 batch 仍为 4；模型收到 8 张两视图图片，调用原 criterion 对这个 concat batch 进行原版 GT 数量归一化，GT/DN 各自正确处理。MERT 模型调用次数是一次，但处理图像量增加，不宣称训练零开销。保留单卡 sequential 作为可选路径；DDP 下显式拒绝 sequential，避免同一步两次 DDP forward 的 reducer 风险。两者在视图 GT 数不等时检测损失归一化不同，正式对比统一 concat。

SECD 与 MERT 无内部变量耦合。组合方法两视图均经过同一个 SECD Backbone。evaluate/test 不创建 MERTBatch、不生成 shifted view、不计算 trajectory 或 MERT loss，只有单次真实 detector forward。

**历史一致性边界：**当前干净版本没有此前 MERT 源码。本实现使用本次明确指定的 late_xywh 参数和公式，已做数值公式测试，但不能据此声称与此前已删除实现的隐含 reduction、BN/DN 行为和整段训练数值逐位一致。

## 5. SECD 的精确定义

保留完整原版 stage：先 `base=Stage_(s+1)(F_s)`，再 `output=base+SECDTransition(F_s)`。SECD 不替换 Stage4/Stage5、不包装重命名 stage、不增加 P2、attention、frequency、wavelet、SPDConv 或 deformable conv。

```text
X_k = X[:, :, row_phase::2, col_phase::2]，四种 phase
μ = mean_k(X_k); R_k = X_k - μ
按 G=8 分组，要求 Cin % G == 0
e_(g,k) = sqrt(mean_(c∈g)(R_(k,c)^2) + 1e-6)
p_(g,k) = softmax_k(e_(g,k) / 0.1)；只沿四个 phase 归一化
κ_g = clamp((4*Σ_k p_(g,k)^2 - 1)/3, 0, 1)
D_g = Σ_k p_(g,k)*R_(g,k); S_g = κ_g*D_g
S = concat_groups(S_g)
E = 1×1 Conv + 原 PResNet 风格 Norm，无额外 activation
α_eff = 0.20*tanh(raw_alpha), raw_alpha_init = 0
SECDTransition(X) = α_eff*E
```

奇数尺寸只在右/下 replicate pad，使输出为 ceil(H/2)×ceil(W/2)。FP16/BF16 的残差能量与 softmax 使用 FP32 计算，输出 evidence 转回输入 dtype。Norm 遵守原 freeze_norm 策略；当前 R18 配置是 BatchNorm。旁路初始化用 fork_rng 保留后续 Encoder/Decoder 的原种子初始化序列。

SECD34 使用 S3 输入，在完整 S4 输出后加旁路；SECD45 使用 S4 输入，在完整 S5 输出后加旁路；SECD345 同时启用，复用同一 SECDTransition 类。当前工程真实原 key 是 `backbone.res_layers.*`，不是示例中的 stages；全部原 key 保持，新增 key 仅在 `backbone.secd_34.*`、`backbone.secd_45.*`。

alpha=0 的首次 forward 与原版一致，但仍计算启用的旁路，不能说 SECD 零 FLOPs。此时 projection 梯度为零是乘法门控的数学结果，raw_alpha 有梯度；非零 alpha 测试验证 projection 和上游特征均获得有限梯度。

## 6. 七份方法 YAML、继承和差异

目录为 `configs/rtdetr/`；下表文件名全部真实存在。每种方法都有独立的 `./output/<同名stem>` 目录。

| 方法 | YAML | 直接父配置 | 相对 DUT Baseline 的方法差异 |
| --- | --- | --- | --- |
| Baseline | rtdetr_r18vd_dut_anti_uav.yml | 官方 rtdetr_r18vd_6x_coco.yml + ../dataset/dut_anti_uav_detection.yml | MERT.enabled=False；SECD.enabled=False |
| MERT | rtdetr_r18vd_dut_anti_uav_mert_late_xywh.yml | DUT Baseline | MERT 开启及第 4 节参数 |
| SECD34 | rtdetr_r18vd_dut_anti_uav_secd_34.yml | DUT Baseline | SECD 开启，transitions=[3to4]，第 5 节参数 |
| SECD45 | rtdetr_r18vd_dut_anti_uav_secd_45.yml | SECD34 | 同参数，transitions=[4to5] |
| SECD345 | rtdetr_r18vd_dut_anti_uav_secd_345.yml | SECD34 | 同参数，transitions=[3to4,4to5] |
| SECD34+MERT | rtdetr_r18vd_dut_anti_uav_secd_34_mert_late_xywh.yml | MERT + SECD34 | 同时启用上述完整 MERT/SECD34；本地重新设 MERT.enabled=True 防止父级 Baseline 关闭开关 |
| SECD345+MERT | rtdetr_r18vd_dut_anti_uav_secd_345_mert_late_xywh.yml | SECD34+MERT | transitions=[3to4,4to5] |

唯一 dataset YAML 为 `configs/dataset/dut_anti_uav_detection.yml`，只设置真实路径、类别数/映射和 test 路径元数据。不创建 DUT 公共 optimizer/dataloader/augmentation 配置。完整递归解析检查已通过：方法间除了 MERT/SECD、include 和输出目录，全部其余字段相同；DUT Baseline 相对官方仅改变数据集相关字段、默认关闭方法开关及输出目录。

完整字段级原始结果见 `dut_resolved_audit.json`，统计检查的代码在 `tools/analyze_dut_models.py`。

## 7. Params / FLOPs 的实际统计

真实 CPU FP32 推理输入 `[1,3,640,640]`，eval、非 deploy，seed=0、PyTorch=2.0.0+cpu、4 线程。仅统计进程内临时关闭 pretrained 下载，训练 YAML 的 pretrained=True 未改变。每种方法独立构建真实模型，执行一次 forward。

**Profiler GFLOPs-LB 是已支持算子的下界，不是完整模型总 FLOPs。**未完整统计 grid_sample、norm、softmax、激活、插值及 SECD reduction/sqrt/tanh 等算子。Conv/Linear GMACs-LB 仅统计实际执行的 Conv2d/Linear 模块，不包含所有 functional attention 运算；不能把下面两列混为完整总量或直接对照论文 GFLOPs。

| 方法 | Params | 新增 Params | Profiler GFLOPs-LB | Conv/Linear GMACs-LB |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 20,083,028 | 0 | 61.151874600 | 30.0069632 |
| MERT | 20,083,028 | 0 | 61.151874600 | 30.0069632 |
| SECD34 | 20,116,309 | 33,281 | 61.257269802 | 30.0593920 |
| SECD45 | 20,215,125 | 132,097 | 61.256969002 | 30.0593920 |
| SECD345 | 20,248,406 | 165,378 | 61.362364204 | 30.1118208 |
| SECD34+MERT | 20,116,309 | 33,281 | 61.257269802 | 30.0593920 |
| SECD345+MERT | 20,248,406 | 165,378 | 61.362364204 | 30.1118208 |

三对 Baseline/MERT、SECD34/SECD34+MERT、SECD345/SECD345+MERT 的实际模型参数、结构、全部权重/输出字节哈希、执行模块路径和已统计算子 FLOPs/MACs 完全相同。因 MERT 不进入 inference graph，总推理复杂度的等价性也来自同一图结构，而非只依赖不完整的 profiler 数字。

每次输出仅有 `pred_logits=[1,300,1]`、`pred_boxes=[1,300,4]`，均有限，无训练轨迹或辅助输出。完整统计环境、各算子计数和推理等价检查在 `dut_model_metrics.json`；说明在 `dut_metrics_summary.md`。Windows PyTorch 2.0 profiler 未计数 FLOPs 的诊断事件个数存在少量记录波动，不将其当作严格等价断言。

## 8. 验证结果与边界

运行 `python -m unittest discover -s tests -v`：**63 项全部通过，13.182 秒**。

| 测试组 | 数量 | 实测覆盖 |
| --- | ---: | --- |
| test_secd.py | 23 | 独立公式参考、uniform/sparse/texture/permutation、奇数尺寸、低精度稳定性、alpha=0 等价及梯度、原 key、预训练兼容、冻结、种子流保持 |
| test_mert.py | 20 | 像素/GT shift、ID、边界 GT、inverse、不同 query 配对、未匹配、轨迹排除 encoder/DN、xy/wh 与小目标权重数值、关闭/eval 零插件行为、concat DN/criterion/backward |
| test_analyze_dut_models.py | 9 | 独立解析、字段公平性 guard、模型等价检查拒绝真实差异、profiler 下界诊断口径 |
| test_dut_integration.py | 10 | 官方 YAML/入口未变、完整 R18 原版参数/key/初始化/640 输出等价、三个 SECD alpha0 完整输出等价、关闭时单步训练更新与原版相同、真实组合模型 DN/反传、eval 单次调用、test 路径适配 |
| test_mert_distributed.py | 1 | 两个 CPU Gloo 进程、真实 R18+SECD34、两步 concat forward+原 criterion+MERT backward、不同 rank GT 数与裁剪、DN、全部参数有限梯度、每步单次 DDP/SECD forward |

测试使用合成输入，无正式训练、无精度/AP 测试。CPU Gloo 测试不使用 CUDA SyncBN，不能替代服务器 NCCL/AMP 的端到端验证。DDP 的 find_unused_parameters 提示来自保留的原版设置，本次未为消除提示而更改它。

## 9. 训练和评估命令

在服务器 `rtdetr_pytorch` 目录执行。下面都是一行命令，统一使用原版训练 AMP、seed=0、三卡原图 batch=12。**同一组 GPU 的实验按顺序运行，不要同时启动这七条。**如果改为单卡或双卡，所有方法保持同样卡数，不自动改 LR/batch。

Baseline：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9911 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml --amp --seed 0
```

MERT：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9912 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_mert_late_xywh.yml --amp --seed 0
```

SECD34：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9913 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_34.yml --amp --seed 0
```

SECD45：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9914 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_45.yml --amp --seed 0
```

SECD345：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9915 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_345.yml --amp --seed 0
```

SECD34+MERT：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9916 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_34_mert_late_xywh.yml --amp --seed 0
```

SECD345+MERT：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9917 tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_345_mert_late_xywh.yml --amp --seed 0
```

推荐顺序：Baseline → MERT → SECD34 → SECD45 → SECD345 → 根据 val 选择 SECD 组合 MERT。不要根据独立 test split 挑选方法。

原版验证集评估，示例 SECD34（FP32）：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_34.yml -r output/rtdetr_r18vd_dut_anti_uav_secd_34/checkpoint.pth --test-only
```

真实 DUT 测试集评估，示例 SECD34（FP32）：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/test_dut.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_secd_34.yml -r output/rtdetr_r18vd_dut_anti_uav_secd_34/checkpoint.pth --split test
```

其他方法同时替换 YAML 和 checkpoint 目录，结构必须匹配。新入口也可用 `--split val`。不添加旧版 `--debug-eval-amp`、`--test-num-workers`；原入口即使传 `--amp` 也不会把原版 evaluate 自动改成 AMP。

原版保存最近 `checkpoint.pth` 与原策略周期快照，打印 best_stat，**不会自动保存 best.pth**。本次没有恢复旧版 best-saving patch；示例 checkpoint.pth 是最近模型，不冒充最佳模型。需要严格测试最佳 epoch 时，用实际保存的该 epoch 快照；若要每轮保存最佳模型，需另做独立改动。

复现配置审计/统计/测试：

```text
python tools/analyze_dut_models.py --resolved-only --output reports/dut_resolved_audit.json
python tools/analyze_dut_models.py --model-only-import --threads 4 --seed 0 --output reports/dut_model_metrics.json
python -m unittest discover -s tests -v
```

## 10. 用户要求的 48 项逐项答复

| # | 项目 | 答复 |
| ---: | --- | --- |
| 1 | 原始 R18 Baseline | configs/rtdetr/rtdetr_r18vd_6x_coco.yml；第 1 节给出实际继承结果 |
| 2 | 原始 train base resize | 640×640，未改 |
| 3 | 原始 multi-scale 完整列表 | [480,512,544,576,608,640,640,640,672,704,736,768,800]，直接继承 |
| 4 | 原始 val/test resize | 640×640，未改 |
| 5 | 原始 eval_spatial_size | Encoder/Decoder 均 [640,640] |
| 6 | 原始 batch | train 4/GPU，val/test 8/GPU；MERT 仅额外创建 shifted 视图 |
| 7 | 原始 optimizer | AdamW，betas=[0.9,0.999]，WD=1e-4，原分组未改 |
| 8 | 原始 LR | Backbone 1e-5，其他 1e-4，未改 |
| 9 | 原始 epoch | 72，未改为 200 |
| 10 | 原始 augmentation | 第 1 节完整顺序/参数，逐项继承，无另加尺寸 cap |
| 11 | DUT dataset/config 文件 | 1 份 dataset YAML + 第 6 节 7 份方法 YAML，复用 CocoDetection |
| 12 | DUT 改哪些字段 | 数据路径、num_classes=1、remap=False、test 路径元数据、独立 output_dir、默认关闭开关 |
| 13 | 修改官方 COCO YAML？ | 否；全部 16 份原有 YAML 原始字节哈希不变 |
| 14 | 修改官方公共 include？ | 否 |
| 15 | 哪些已有 Python 被修改 | yaml_config.py、presnet.py、det_engine.py、det_solver.py，仅四处；新增文件见第 3 节 |
| 16 | 每项修改为什么必要 | 配置隔离、旁路接入/兼容、训练插件分支、配置传递；第 3 节逐文件说明 |
| 17 | 关闭模块严格等价？ | 实测完整参数/key/初始化/输出一致；关闭时单步训练更新一致；无额外 Tensor 算子/forward |
| 18 | 原 PResNet state_dict key | 原 backbone.res_layers.* 等 key 完整保留；只增加 secd_34/45 前缀 |
| 19 | S3/S4/S5 channels | 128/256/512 |
| 20 | S3/S4/S5 strides | 8/16/32 |
| 21 | SECD34 插入位置 | 完整 S4 stage 输出之后，加来自其 S3 输入的旁路 |
| 22 | SECD45 插入位置 | 完整 S5 stage 输出之后，加来自其 S4 输入的旁路 |
| 23 | SECD 公式与要求一致？ | 是，包含分组 energy、四 phase softmax、指定 κ、κD、1×1 Conv+Norm；独立参考数值测试通过 |
| 24 | SECD alpha 初始化 | raw_alpha=0，alpha_eff=0.20*tanh(raw_alpha)=0 |
| 25 | SECD unit tests | 23 项全通过；完整模型 alpha=0 等价另在 integration 测试覆盖 |
| 26 | MERT shift 实现 | 输入实际尺度上非零 ±1 像素整数平移、零填充、不环绕 |
| 27 | MERT GT shift | normalized cxcywh 同步平移，stable origin_gt_id，边界裁剪/排除一致性 |
| 28 | MERT Hungarian 配对 | 两视图独立 final-layer 原 matcher，按 GT ID 找 matched queries |
| 29 | MERT inverse alignment | cx/cy 减 dx/W、dy/H，w/h 保持，不裁剪预测 |
| 30 | MERT trajectory | B0→B1→B2 的最后两次 decoder update；排除 encoder proposal 和 DN |
| 31 | MERT loss | 第 4 节明确 SmoothL1 beta=.01，xy=1、wh=.25、weight=.10、小目标权重、N*K 分母 |
| 32 | MERT eval 全关闭？ | 是，不创建 shifted view，不算轨迹/损失，只有单次 detector forward |
| 33 | MERT unit tests | 20 项全通过，另有完整模型 integration 与两进程 DDP 测试 |
| 34 | Baseline Params/FLOPs | 20,083,028；61.151874600 GFLOPs-LB，不是完整总 FLOPs |
| 35 | MERT Params/FLOPs | 与 Baseline 相同，实际结构/计数检查通过 |
| 36 | SECD34 Params/FLOPs | 20,116,309；61.257269802 GFLOPs-LB |
| 37 | SECD45 Params/FLOPs | 20,215,125；61.256969002 GFLOPs-LB |
| 38 | SECD345 Params/FLOPs | 20,248,406；61.362364204 GFLOPs-LB |
| 39 | SECD+MERT Params/FLOPs | 34+MERT 与 SECD34 相同；345+MERT 与 SECD345 相同 |
| 40 | 最终新增 YAML | 第 6 节 7 份方法 + 1 份 dataset，全部真实存在 |
| 41 | 每份 YAML 继承谁 | 第 6 节逐文件父配置表 |
| 42 | 每份与 Baseline 差哪些字段 | 仅方法 MERT/SECD 字段、include/output_dir；完整 resolved diffs 已保存 |
| 43 | 推荐训练命令 | 第 9 节 7 条一行命令，前三项优先 |
| 44 | 推荐测试命令 | 第 9 节 native val 和新增 true test 两条，均原版 FP32 |
| 45 | 有 800 固定输入？ | 无 DUT 800 固定输入；原版随机 multi-scale 中有 800，按要求保留 |
| 46 | 有 960 DUT 尺寸？ | 无 |
| 47 | 恢复了不应恢复的旧实验？ | 否，只有本次需要的 MERT/SECD；无旧模块/分辨率/eval AMP/test-worker 补丁 |
| 48 | 已知风险 | 第 11 节明确列出真实数据增强内存、训练成本/BN、历史还原边界、checkpoint 和测试环境限制 |

## 11. 已知风险与后续边界

1. **原版增强对 DUT 原始大图的风险。**实读 train 最大 5616×3744，val 最大 4288×2848，test 最大 1920×1080。RandomZoomOut 在 Resize 之前执行，原版默认可放大至 4 倍边长；最大训练图的中间画布可约 3.36 亿像素，存在 CPU/RAM/PIL decompression-bomb 等问题。为不偷改增强，本次不预缩图、不降低倍率、不关闭 PIL 安全机制；若服务器出现相关报错，需要独立批准数据增强鲁棒性适配，并在所有方法统一采用。
2. **训练资源/统计改变来自方法本身。**MERT 多一个视图，concat batch 的图像量翻倍，显存/计算增加，BatchNorm 和 DN batch 行为也随视图增加。保留原数据 loader batch 不等于训练处理量不变；不自动减 batch、增 LR 或使用额外梯度累积。
3. **历史结果不能直接复用。**本次原版 72e/640 协议不同于旧 200e/高分辨率协议；旧 AP 不是这套新 Baseline 的实测结果。MERT 原已删除实现的隐含 reduction 无法从当前代码证实，本报告不宣称逐位恢复旧训练轨迹。
4. **初始化门控。**alpha=0 时 projection 初始梯度为零，raw_alpha 可学习；非零之后旁路学习。SECD 继承原 Backbone LR=1e-5，可能起效较慢，这是保留原协议的结果，没有擅自加旁路 LR。
5. **Checkpoint 范围。**原 Backbone 预训练加载允许仅新增 SECD key 缺失并检查其他错配；`-r` 完整训练恢复仍是原版严格架构/optimizer 语义，不能把 Baseline checkpoint 直接作为 SECD 的完整 resume。需要同架构 resume；跨架构初始化若使用原版 `-t`，它不是完整恢复，必须单独说明。原版不自动保存 best.pth。
6. **验证精度公平优先。**没有恢复 eval AMP、测速补丁或 test-worker CLI。所有方法验证/真实 test 使用原版 FP32，worker=4。MERT 训练中仍有 ID 配对 CPU 同步开销，未作为本任务偷偷更改原训练基础实现。
7. **测试覆盖不是服务器实跑承诺。**63 项测试是 CPU 与合成输入；真实 CUDA AMP/NCCL、完整数据增强、服务器多进程 DataLoader 的环境行为及最终 AP 仍需服务器验证。保留的原版 find_unused_parameters=True 可能有 warning/遍历成本，本次不改。
8. **复杂度计数有限。**当前报告提供真实 Params 和明确标注的 FLOPs/MACs 下界，不声称拥有完整 FLOPs、训练速度、峰值显存或检测精度结果。
