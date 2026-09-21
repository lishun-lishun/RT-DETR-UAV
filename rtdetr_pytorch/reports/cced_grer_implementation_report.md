# CCED-34 / GRER-34 实现与验收报告

## 结论

已在原始 PResNet18-d 的 stride 8 -> stride 16 transition 上实现可通过 YAML 独立开关的 CCED-34、GRER-34 及二者并联组合。原 Stage4 完整保留，Stage5 接收增强后的 F4；HybridEncoder、Decoder、Matcher、Loss、数据和训练协议均未改动。默认 `alpha_init=0`，所以开启模块后的初始模型也与 Baseline 输出逐元素相同。

## 1—4：原始 Backbone 审计

1. PResNet18 实现在 `src/nn/backbone/presnet.py`，主体类为 `PResNet`，基本残差块为 `BasicBlock`。
2. 使用真实 `1x3x640x640` 前向和 stage hook 得到：Stem 输出 `[1,64,320,320]`；S2=`res_layers[0]` 输出 `[1,64,160,160]`、stride 4；S3=`res_layers[1]` 输出 `[1,128,80,80]`、stride 8；S4=`res_layers[2]` 输出 `[1,256,40,40]`、stride 16；S5=`res_layers[3]` 输出 `[1,512,20,20]`、stride 32。
3. HybridEncoder 接收的 P3/P4/P5 channel 分别是 C3=128、C4=256、C5=512，stride 为 8/16/32。
4. stride 8 -> 16 真正发生在 `res_layers[2]` 的第一个 `BasicBlock`。新旁路读取完整 S3 的输入 F3，同时保留并执行完整原 S4 得到 `F4_base`。

## 5—7：文件与必要修改

5. 新增 Python 文件：
   - `src/nn/backbone/backbone_plugins/cced.py`
   - `src/nn/backbone/backbone_plugins/grer.py`
   - `tests/test_cced_grer.py`
   - `tools/benchmark_cced_grer.py`
6. 修改原 Python 文件：
   - `src/nn/backbone/presnet.py`：仅增加配置注入、模块构造、pretrained 白名单和一个专用 forward 路径。
   - `src/nn/backbone/backbone_plugins/__init__.py`：导出两个新模块。
   - `tools/analyze_dut_models.py` 与对应测试：让既有公平性审计识别 CCED/GRER 的默认关闭项和方法开关。
7. 这些修改分别用于模块注册、在正确 transition 并联挂载、保持 pretrained/state_dict 兼容，以及防止既有公平性检查把“新增但关闭的开关”误判为训练协议变化。没有修改 HybridEncoder、AIFI、CCFF、Query Selection、Decoder、Matcher、Loss 或 DataLoader。

## 8—16：CCED 实现

8. 真实输入 F3 为 `[B,128,80,80]`，输出 evidence 为 `[B,256,40,40]`。
9. 2x2 非重叠 phase split 后堆叠为 `[B,128,4,40,40]`；奇数 H/W 时只在右侧/底部做 replicate pad，避免 silent mismatch。
10. 默认 G=8，group residual reshape 为 `[B,8,16,4,40,40]`。若 C 不能被请求组数整除，按 8、4、2、1 选择不大于请求值的最大合法组数，不截断 channel。
11. `q = e / (sum_phase(e)+eps)`，随后做一次数值再归一化以严格满足 phase 和为 1；未使用 Softmax。
12. `c = exp(mean_group(log(q+eps)))`，即跨 group 几何平均。
13. `c_hat = c/(sum_phase(c)+eps)`，再数值归一化保证和为 1。
14. `D = sum_phase(c_hat * R)`，输出 `[B,128,40,40]`。
15. 使用 PResNet 风格 `1x1 Conv + BatchNorm`，无 activation，投影到 256 channel。
16. `alpha_c = 0.20*tanh(raw_alpha)`；默认 `raw_alpha=0`，也支持 YAML 设为 0.01 等非零小值。

## 17—26：GRER 实现

17. 局部均值为 `avg_pool2d(X,3,stride=1,padding=1,count_include_pad=False)`；后者使常数图边界仍保持常数，再计算 `R=X-local_mean`。
18. reshape 为 `[B,8,16,H,W]`，计算每组 RMS deviation，再跨 8 组平均为 `[B,1,H,W]` 的 local score。
19. median 在每个 sample 的空间维单独计算，形状 `[B,1,1,1]`，不跨 batch/channel/dataset。
20. MAD 同样为每图空间中位数；`mad_safe=clamp_min(MAD,1e-6)`。
21. `z=(score-median)/mad_safe`。
22. `gate=sigmoid((z-2.0)/1.0)`，范围为 `[0,1]`。
23. rare feature 严格使用 `gate*X`，不是 `gate*R`。
24. 奇数尺寸先右/下 replicate pad，再用 `AvgPool2d(2,2)`；未用 MaxPool。
25. 使用无 activation 的 `1x1 Conv + BatchNorm` 将 128 投影到 256 channel。
26. `alpha_r = 0.20*tanh(raw_alpha)`，默认 `raw_alpha=0`。

## 27—28：联合与 Stage5

27. 联合模式严格并联：`F4 = F4_base + alpha_c*E_cced(F3) + alpha_r*E_grer(F3)`。两个模块读取同一个原始 F3，没有串联、Concat 或额外融合层。
28. `res_layers[3]` 直接读取上述增强后的 F4，因此 `F5=Stage5(F4)`。测试用非零 alpha 手工复算 F3/F4/F5，与实际 Backbone 输出一致。

## 29—36：测试结果

29. Baseline equivalence：用只读 `git show HEAD` 载入修改前 PResNet，固定 seed/权重/input/eval，P3/P4/P5、Encoder、Decoder、final boxes、final logits 全部逐元素一致，`atol=rtol=0`。
30. CCED enabled 且 alpha=0：Backbone 和最终预测相对 Baseline 的最大差值为 0。
31. GRER enabled 且 alpha=0：最大差值为 0；联合模式两个 alpha=0 时也为 0。
32. CCED 单测覆盖：相同 phase 零 evidence、所有 group 共同支持、单 group 支持被几何共识抑制、phase permutation 等变、奇数尺寸、group fallback、q/consensus 精确归一化及禁止 Softmax。
33. GRER 单测覆盖：flat map/MAD=0 安全、single sparse 比 repeated texture 更稀有、per-image 统计、严格 `gate*X`、median/MAD 可微。
34. 三步梯度实测（依次为 raw-alpha/projection/F3 grad norm）：
   - CCED：step0 `1.362e-3 / 0 / 1.470e-2`；step1 `8.959e-3 / 4.077e-6 / 1.470e-2`；step2 `6.647e-3 / 2.121e-5 / 1.470e-2`。
   - GRER：step0 `7.975e-3 / 0 / 1.483e-2`；step1 `5.317e-5 / 2.938e-5 / 1.483e-2`；step2 `3.466e-3 / 2.202e-5 / 1.483e-2`。
   - 结论：zero alpha 时 projection 第一步零梯度是预期数学行为；raw alpha 更新后 projection 从第二步开始学习，F3 始终有有限非零梯度。
35. CUDA AMP 下对 CCED、GRER、联合三种模式均执行了真实 `1x3x640x640` Backbone forward/backward；只把 energy/log/median/MAD 等统计局部提升到 FP32，输出再 cast 回 feature dtype。
36. 所有 AMP loss、输出、输入梯度和已产生的参数梯度均为 finite，无 NaN/Inf。新功能测试 16/16、原 Backbone 插件回归 13/13、SECD 回归 23/23、公平性审计 15/15 通过。本机 torchvision 0.25 已移除旧项目依赖的 `torchvision.datapoints`，因此依赖该旧接口的 4 个既有数据/MERT 测试无法导入；这不是本次模块运行失败，服务器原 torch2.0.1 环境仍使用项目原依赖组合。

## 37—38：权重兼容性

37. 原 pretrained 下载/加载路径保持不变。开启新模块时只允许 `cced_34.*`、`grer_34.*` 为 missing keys；任何原 stage missing 或 unexpected key 都会抛错。
38. `conv1.*`、`res_layers.*` 等原 key 和原始张量初始化不变；新增 key 只位于 `cced_34.*`、`grer_34.*`（完整 detector 中为 `backbone.cced_34.*`、`backbone.grer_34.*`）。forked RNG 保证同 seed 下新增分支不改变 Encoder/Decoder 的初始化序列。

## 39—43：参数、计算量和推理速度

协议：batch=1、640x640、AMP、warmup=100、iterations=500、`torch.cuda.Event`；本地设备 RTX 5060 Ti，PyTorch 2.10.0+cu128。

| 模型 | Params | Delta Params | profiler counted GFLOPs | Delta GFLOPs | Mean ms | Median ms | P95 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 20,083,028 | 0 | 112.0066 | 0 | 11.1789 | 10.6009 | 15.2030 | 89.45 |
| CCED34 | 20,116,309 | 33,281 | 112.2168 | 0.2102 | 11.8254 | 11.0963 | 17.3872 | 84.56 |
| GRER34 | 20,116,309 | 33,281 | 112.2168 | 0.2102 | 11.9357 | 11.0257 | 17.7646 | 83.78 |
| CCED34+GRER34 | 20,149,590 | 66,562 | 112.4270 | 0.4204 | 12.7650 | 11.6005 | 19.1343 | 78.34 |

GFLOPs 是 PyTorch profiler 对已计数算子的下界，不应当冒充精确总 FLOPs；sqrt/log/median/MAD/sigmoid 等部分 elementwise 算子未由 profiler 分配 FLOPs。延迟是本机实现验收数据，A30 服务器应使用同一工具重新测量，不能直接套用本机数值。

## 44—45：YAML 与公平性

44. 新增：
   - `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34.yml`
   - `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_grer34.yml`
   - `configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34_grer34.yml`
45. 三者相对 Baseline 只修改 `output_dir` 和对应的 `CCED.*`/`GRER.*` 字段。Dataset、输入、多尺度、增强、batch、200 epoch、每 10 epoch checkpoint、Optimizer、LR、Scheduler、EMA、AMP CLI、pretrained、Encoder、Decoder、query、Matcher、Loss 均继承相同设置；MERT 与 SECD 明确保持关闭。

## 46：推荐训练命令（单卡，一行一条）

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_grer34.yml --amp --seed 0
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34_grer34.yml --amp --seed 0
```

公平对比 Baseline：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml --amp --seed 0
```

## 47：推荐测试与复测命令

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34.yml -r output/rtdetr_r18vd_dut_anti_uav_cced34/best.pth --test-only --amp
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_grer34.yml -r output/rtdetr_r18vd_dut_anti_uav_grer34/best.pth --test-only --amp
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python tools/train.py -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34_grer34.yml -r output/rtdetr_r18vd_dut_anti_uav_cced34_grer34/best.pth --test-only --amp
python -m unittest discover -s tests -p "test_cced_grer.py" -v
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_cced_grer.py --amp --warmup 100 --iterations 500 --output reports/cced_grer_benchmark_a30.json
```

## 48：当前风险与观察项

1. alpha 默认严格为零，projection 第一步没有梯度；三步检查确认第二步能开始学习，但训练时仍应观察 alpha 是否长期停在零附近。
2. CCED 的几何平均会主动惩罚“仅少数组支持”的 phase；若真实小目标只激活少数语义 channel group，可能过度抑制。应观察 consensus entropy、是否全均匀/过尖和 evidence/base ratio。
3. GRER 的 median/MAD 完整可微但梯度稀疏；MAD 接近零时虽然数值有限，z 可能很大并使 sigmoid 饱和。第一轮不 detach、不自动改公式，只通过 debug 观察 MAD、z 与 gate 分布。
4. BatchNorm 在单卡 batch=16 下合理，但分支统计是新参数，不能期望仅加载 pretrained 就已有良好 running statistics。
5. 本地只完成结构、数值、性能和兼容验收，没有替代 200 epoch 精度实验；AP、AP50、AP75、AP-small、AR100、AR-small 是否提升必须由四组同协议训练决定。
6. 本机速度不能代表 A30；应在服务器用同一 100/500 CUDA Event 协议复测。
7. `debug=true` 会执行分位数、标量转 Python 和打印，刻意只用于诊断；正式训练必须保持 `debug=false`。
