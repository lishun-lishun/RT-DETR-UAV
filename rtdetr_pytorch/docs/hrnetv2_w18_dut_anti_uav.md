# RT-DETR HRNetV2-W18 接入报告

## A. HRNet 实现

- 新增 `src/nn/backbone/hrnet.py`：独立 `HRNetV2W18` Backbone provider。
- 修改 `src/nn/backbone/__init__.py`：导入并注册 HRNet。
- Backbone 注册沿用 `src/core/yaml_utils.py` 的 `@register`；模型仍由
  `RTDETR.backbone` 注入，不改变 `PResNet`。
- 标准 W18 结构：Stage2 为 1 个二分支模块（18/36），Stage3 为 4 个
  三分支模块（18/36/72），Stage4 为 3 个四分支模块
  （18/36/72/144）。每个模块包含并行分支及跨分辨率融合。
- 实现的 1830 个 Backbone state-dict 项与成熟 timm `hrnet_w18` 的
  键名和尺寸逐项一致；加载同一官方权重后，三层输出与 timm 在逐元素
  比较中完全相等（3 层最大绝对误差均为 0）。
- `pretrained: true` 使用 ImageNet 文件
  `hrnetv2_w18-8cb57bb9.pth`；`pretrained_path` 可指定服务器本地文件。
  实际官方权重加载结果：matched=1830、missing=0、unexpected=0，忽略的
  126 项全部属于分类头。
- 来源：Microsoft HRNet Image Classification 与其 timm 迁移版本；许可
  声明保留在源码文件头。

离线服务器可把权重放到
`rtdetr_pytorch/weights/hrnetv2_w18-8cb57bb9.pth`，并将 include YAML 的
`pretrained_path` 设置为该相对路径；文件不存在时会明确报错，不会静默
退回随机初始化。

## B. Feature 输出

输入 `1×3×640×640` 的实测结果：

| 检测层 | Shape | Stride | Channels |
|---|---:|---:|---:|
| P3 | `1×36×80×80` | 8 | 36 |
| P4 | `1×72×40×40` | 16 | 72 |
| P5 | `1×144×20×20` | 32 | 144 |

Stride-4 分支始终保留在 HRNet 内参与融合，但不送入 HybridEncoder。

## C. HybridEncoder

- PResNet18 baseline `in_channels`: `[128, 256, 512]`
- HRNetV2-W18 `in_channels`: `[36, 72, 144]`
- 两者 `feat_strides`: `[8, 16, 32]`
- `hidden_dim`、AIFI、CCFF、encoder 层数与其余 Encoder 配置未修改。
- HRNet 后没有增加 36→128、72→256、144→512 额外投影；直接使用原
  HybridEncoder 自带的输入投影。

## D. 训练参数对比

| 参数 | Baseline | HRNet |
|---|---:|---:|
| 主学习率 | 0.0003 | 0.0003 |
| Backbone 学习率 | 0.00003 | 0.00003 |
| Epoch | 200 | 200 |
| 每卡 Batch | 16 | 16 |
| 三卡 Global Batch | 48 | 48 |
| 验证分辨率 | 640×640 | 640×640 |
| 训练 Resize | 640×640 | 640×640 |
| 训练 multi-scale | 480–800（原列表） | 完全相同 |

配置解析后的完整差异审计只得到：`RTDETR.backbone`、
`HRNetV2W18.pretrained/pretrained_path`、`HybridEncoder.in_channels`、
`output_dir` 和 include 元数据。优化器、scheduler、warmup、EMA、AMP、
数据增强、DataLoader、Decoder、Matcher 和 Loss 均继承 baseline。

## E. 三卡配置

- `CUDA_VISIBLE_DEVICES=1,2,3`
- `nproc_per_node=3`
- 实际 HRNet 单实验启动命令：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9909 tools/train.py -c configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml --amp --seed 0
```

三卡 DDP 短测试命令（不会读取数据或进行完整训练）：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=9919 tools/validate_hrnetv2_w18.py --ddp-smoke
```

## F. 批量训练顺序

`tools/train_all_dut_modules_3gpu.sh` 的顺序为：

1. HRNetV2-W18（新增）
2. PDR3（原第 1）
3. PDR34（原第 2）
4. PDR34-NoGate（原第 3）
5. Baseline（原第 4）
6. BDPD（原第 5）
7. MSDConv（原第 6）
8. P4-FADC
9. DEConv
10. SECD345+MERT
11. HSDR-A
12. BAFR+HCBR
13. BAFR
14. BDPD+MSDConv
15. CCED34
16. GRER34
17. CCED34+GRER34
18. PHSB
19. HCBR
20. RFAConv
21. SECD34+MERT
22. AKConv
23. SECD345
24. MERT-late-xywh
25. P4-WTConv
26. SRFD
27. DRB
28. P0-SRFD
29. P3-SECD34
30. SECD45
31. SECD34
32. HSDR-B
33. P1-DEConv
34. P2-DCNv4

以上 8–34 是本机验收 dry-run 生成的真实顺序。脚本保留了原来的
`sort | shuf` 随机尾部策略，所以服务器下一次启动时尾部可能重新排列；
脚本会在任何训练开始前打印该次运行的完整 1–N 顺序。

## G. output skip 与 dry-run

脚本通过项目自身 `load_config` 解析继承配置，再应用与训练命令相同的
`--output-dir` 覆盖。仅检查当前实验的最终完整路径：路径存在即 SKIP，
不存在才 RUN；不检查 checkpoint、指标或完成标记，也不删除旧结果。

```bash
bash tools/train_all_dut_modules_3gpu.sh --dry-run
```

dry-run 会打印 `Total configs / Will run / Will skip` 及每个实验的状态和
输出路径，并在创建日志目录、导出 CUDA 环境或调用 torchrun 之前退出。
本机实际验收结果为 `Total=34, Will run=34, Will skip=0`（本机尚无该批次
output）；运行前后 queue output 与 log 目录均不存在，证明 dry-run 没有
创建目录。服务器上的 RUN/SKIP 数量则由服务器实际 output 目录决定。

## H. 验证结果

| 检查 | 结果 |
|---|---|
| 标准结构及 640 Backbone forward | PASS |
| HRNet + 原 HybridEncoder + 原 Decoder forward | PASS |
| Stem、Stage1–4 与全部参数 backward | PASS |
| CUDA AMP forward/backward | PASS |
| 真实 ImageNet pretrained load | PASS（1830/1830 Backbone 项） |
| 三卡 CUDA DDP startup | 需在三卡服务器运行上述短测试命令 |

复杂度是 640×640 下 Conv/Linear hook 的可复现下界（不含 norm、activation、
interpolate、softmax 和 functional attention），FLOPs 按 2×MAC 计算：

| 模型 | Backbone Params | Whole Params | Backbone MACs | Whole MACs |
|---|---:|---:|---:|---:|
| PResNet18 baseline | 11,199,968 | 20,083,028 | 16.866G | 30.007G |
| HRNetV2-W18 | 9,562,260 | 18,280,456 | 26.792G | 39.669G |

HRNet 参数更少，但高分辨率并行分支使计算量更高，这是标准 HRNet 的预期
特性，不是额外检测层或插件造成的。
