# DUT-Anti-UAV Baseline vs HRNetV2-W18 公平性审计

> 本报告由只读审计生成；没有训练模型、修改配置或改写 checkpoint。

## 输入尺寸

| Stage | Baseline | HRNet | Same? |
|---|---|---|---|
| Train base resize | `640×640` | `640×640` | YES |
| Train multi-scale | `[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]` | `[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]` | YES |
| DataLoader train batch | `NOT RUN (pycocotools unavailable)` | `NOT RUN (pycocotools unavailable)` | YES |
| Actual train backbone inputs | `[[1, 3, 800, 800], [1, 3, 640, 640], [1, 3, 672, 672], [1, 3, 736, 736], [1, 3, 736, 736], [1, 3, 576, 576], [1, 3, 736, 736], [1, 3, 608, 608]]` | `[[1, 3, 800, 800], [1, 3, 640, 640], [1, 3, 672, 672], [1, 3, 736, 736], [1, 3, 736, 736], [1, 3, 576, 576], [1, 3, 736, 736], [1, 3, 608, 608]]` | YES |
| Val batch | `NOT RUN (pycocotools unavailable)` | `NOT RUN (pycocotools unavailable)` | YES |
| Test batch | `NOT RUN (pycocotools unavailable)` | `NOT RUN (pycocotools unavailable)` | YES |
| eval_spatial_size | `[640, 640]` | `[640, 640]` | YES |
| P3/P4/P5 stride | `[8,16,32]` | `[8,16,32]` | YES |

- Validation/Test 均固定为 **640×640**。
- 800 仅是训练 multi-scale 候选值，不是固定评测分辨率。
- 960 不存在于当前 Baseline/HRNet resolved 训练或评测配置。
- `best_stat` 与 `tools/train.py --test-only` 使用 Val；`tools/test_dut.py --split test` 才显式切换到 Test。

## Val/Test 结果

| Model | best_stat | rerun Val | Test | Test-Val |
|---|---:|---:|---:|---:|
| PResNet18 | PENDING | PENDING | PENDING | PENDING |
| HRNetV2-W18 | PENDING | PENDING | PENDING | PENDING |

## 完整 COCO 指标

### PResNet18 / VAL

PENDING：本机没有对应 checkpoint。

### PResNet18 / TEST

PENDING：本机没有对应 checkpoint。

### HRNetV2-W18 / VAL

PENDING：本机没有对应 checkpoint。

### HRNetV2-W18 / TEST

PENDING：本机没有对应 checkpoint。

## 数据分布摘要

| Split | Images | Objects | Obj/Image mean | Box area median | Relative area median | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5200 | 5246 | 1.008846 | 969.000000 | 0.000473 | 2723 | 1860 | 663 |
| val | 2600 | 2620 | 1.007692 | 946.000000 | 0.000460 | 1401 | 892 | 327 |
| test | 2200 | 2245 | 1.020455 | 1674.000000 | 0.000911 | 848 | 836 | 561 |

## 公平性结论

- Resolved config 仅有允许差异：**YES**
- Train/Val/Test、增强、batch、epoch、优化器 YAML、LR、scheduler、EMA、Detector、Matcher、Loss：相同。
- PResNet18 与 HRNetV2-W18 均使用各自 ImageNet 预训练权重。
- 实际优化器参数语义一致：**NO**
- 当前 HRNet BatchNorm 名称未命中 `norm` 正则，归一化参数使用了普通 Backbone 的 weight decay。
- `best_stat` 来自 `val_dataloader`；原始及当前 `tools/train.py --test-only` 也评估 Val。
- 真正 Test 由 `tools/test_dut.py --split test` 或本审计工具显式切换 split。
- 当前公平性等级：**MOSTLY FAIR**。
- 当前结果可做 Backbone 对比，但不能表述为严格控制全部优化器语义的 STRICTLY FAIR 实验。

## 数据泄漏检查

- 文件名交集：`{'train_x_val': 2600, 'train_x_test': 2200, 'val_x_test': 2200}`
- SHA256 内容交集：`{'train_x_val': 0, 'train_x_test': 1, 'val_x_test': 0}`
- SHA256 重复明细：`{'train_x_val': [], 'train_x_test': [{'sha256': '29b76668151334817e45f5b52a9da89dadd52a3d91a2fbee01b40311b5559c51', 'train_files': ['00259.jpg'], 'test_files': ['01374.jpg']}], 'val_x_test': []}`

