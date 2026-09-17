# DUT 实验配置公平性与模型统计

7 份 DUT YAML 的完整继承解析检查通过。Baseline 相对官方 R18 只改变数据路径、类别数及 COCO 类别映射、关闭的新方法开关、输出目录，并新增不参与训练的 test_dataset 路径元信息。所有方法相对 DUT Baseline 只改变 MERT/SECD 字段及 include/输出目录；epoch、batch、优化器、LR、scheduler、增强、分辨率、预训练策略、EMA、AMP、criterion/decoder 等其余字段完全相同。

## 统计口径

真实模型源码、真实 CPU forward，无依赖或模型 stub。使用 PyTorch `2.0.0+cpu`，线程数 4，seed 0，随机初始化，单张 `[1,3,640,640]` FP32 输入，`model.eval()`，非 deploy，关闭 AMP。仅在统计进程的内存副本关闭 backbone pretrained，避免下载；训练 YAML 的 pretrained=True 未改变。每种方法独立子进程，避免全局 registry 残留污染。

显式 `--model-only-import` 选择性导入实际 R18、Encoder、Decoder 源码，绕开不相关的 COCO/RegNet eager imports；这不验证数据集依赖或完整训练环境。统计范围是一次 detector forward，不包括数据增强、postprocessor、criterion 或 MERT 训练开销。

下列 GFLOPs-LB 是 PyTorch profiler 已支持算子的 FLOPs 下界，**不是完整模型总 FLOPs**。本机计入 conv2d、mm、bmm、addmm、add、mul；没有完整计入 grid sampling、norm、softmax、激活、插值及 SECD 的 reduction/sqrt/tanh 等运算。Conv/Linear GMACs-LB 则只统计实际执行的 Conv2d/Linear 模块，不含 functional attention projection；只有这一 MACs 指标采用 `FLOPs=2×MACs` 口径。两种下界不能当作完整复杂度直接比较其他论文数值。

| 方法 | 参数量 | Profiler GFLOPs-LB | Conv/Linear GMACs-LB |
| --- | ---: | ---: | ---: |
| Baseline | 20,083,028 | 61.151874600 | 30.0069632 |
| MERT | 20,083,028 | 61.151874600 | 30.0069632 |
| SECD34 | 20,116,309 | 61.257269802 | 30.0593920 |
| SECD45 | 20,215,125 | 61.256969002 | 30.0593920 |
| SECD345 | 20,248,406 | 61.362364204 | 30.1118208 |
| SECD34+MERT | 20,116,309 | 61.257269802 | 30.0593920 |
| SECD345+MERT | 20,248,406 | 61.362364204 | 30.1118208 |

全部参数均 requires_grad=True。SECD34、45、345 分别新增 33,281、132,097、165,378 个参数。SECD34 与 SECD45 的 projection MACs 恰好相同：更深处空间面积下降 4 倍、Cin×Cout 增加 4 倍。两者 profiler FLOPs 下界的微小差异来自被 profiler 部分计入的 SECD elementwise 运算；并非总 FLOPs 差值。

## 推理等价检查

Baseline↔MERT、SECD34↔SECD34+MERT、SECD345↔SECD345+MERT 三对均通过独立实际模型构建检查：参数量、可训练参数量、模型结构、全部 state_dict tensor、输出 tensor、模块执行顺序、Conv/Linear MACs 下界、profiler FLOPs 下界以及已计数算子完全一致。state/output 使用全部 tensor 字节的 SHA256 比对，没有只比 shape 或复用同一个模型。

每次评估仅执行一次 detector forward，返回且只返回 `pred_logits=[1,300,1]`、`pred_boxes=[1,300,4]`，均有限值，没有训练轨迹或辅助输出。此随机初始化样本下，alpha_init=0 的三个 SECD 方法输出也与 Baseline 逐字节相同。

Profiler 未计入 FLOPs 的全部 aten 事件也保存在 JSON 中供审计，但其事件计数不作为严格等价断言：同一 SECD345 配置重复运行已复现 PyTorch 2.0 Windows profiler 的 `aten::div_` 4/5 次及 `aten::upsample_nearest2d` 2/3 次记录波动，而模型权重、输出、已计数算子与 FLOPs 相同。最终保存的三对样本中诊断事件列表也相同，但不将此偶然一致过度解释为全部运算完整计数。

## 复现与测试

从 `rtdetr_pytorch` 目录运行：

```text
python tools/analyze_dut_models.py --resolved-only --output reports/dut_resolved_audit.json
python tools/analyze_dut_models.py --model-only-import --threads 4 --seed 0 --output reports/dut_model_metrics.json
python -m unittest discover -s tests -p test_analyze_dut_models.py -v
```

9 个工具 guard tests 通过，覆盖 fresh loader accumulator、公平性允许/禁止字段、独立配对值校验、输出/模块执行路径差异拒绝及 profiler 未计数事件的诊断口径。这里没有启动训练、下载权重、修改官方 YAML 或提供准确率/测速结果。

完整原始结果见 `dut_model_metrics.json`；其中包含每份 YAML 的字段级 resolved differences、训练协议、每个实际被计数/未计数算子、全部等价检查与统计环境。更全面的模型总 FLOPs 需要支持实际 grid sampling、attention 和 SECD 运算的补充计数器，当前不声称已获得。
