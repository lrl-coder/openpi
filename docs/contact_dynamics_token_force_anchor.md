# Contact Dynamics Token: force encoder physical anchoring and asymmetric distillation

本文档记录 force-guided pi0 模块的一次关键修改：把原来的双分支 cosine semantic alignment，改成带物理锚点的 Contact Dynamics Token 训练目标。

当前两阶段方案中，本模块在阶段一完成训练；阶段二将其冻结，并由
[Force Reflex Adapter](force_reflex_adapter.md) 直接复用最新力历史对应的 Contact Dynamics Token，
把物理接触表示转换为关节空间连续修正。

## 背景问题

之前的 force semantic 分支使用：

```text
loss = 1 - cosine(query_feature, force_feature)
```

其中 `query_feature` 来自 prefix 中的 semantic query，`force_feature` 来自 TCN(force history)。两个分支都可训练，并且都做 L2 normalize。

这个目标在小数据、单任务、prompt 高度相似的场景里有一个非常便宜的坏解：两个 encoder 对所有样本输出几乎相同的单位向量。这样 cosine 会快速接近 1，alignment loss 很低，但 feature 失去区分不同接触状态的能力。训练中看到的 `query_batch_std_mean` 和 `force_batch_std_mean` 接近 0 就是这个现象。

## 新设计

新模块把 force feature 定义为一个轻量的 contact dynamics representation。核心思想是：

```text
force branch must explain physical force statistics first;
the prefix semantic query only distills this physically grounded representation.
```

也就是说，force branch 不再只是被另一个可漂移的 embedding 拉近，而是必须预测未来 force 的可解释物理 summary。prefix semantic query 仍然作为 alignment student，但它只学习一个被物理目标锚定住的 stop-gradient teacher。

## 1. Physical anchoring

代码位置：

```text
src/openpi/models/pi0.py
```

新增 head：

```text
force_physical_summary_out: force_raw_feature -> 5 * force_dim
```

TCN force encoder 先输出未归一化的 `force_raw_feature`：

```text
force_history_global -> TCN -> mean pooling -> force_semantic_out -> force_raw_feature
```

然后预测未来 action horizon 内 force_targets 的 summary：

```text
summary = concat(
    mean(force_targets),
    std(force_targets),
    max(abs(force_targets)),
    force_targets[-1] - force_targets[0],
    mean(abs(force_targets)),
)
```

loss：

```text
L_phys = MSE(force_physical_summary_out(force_raw_feature), stopgrad(summary))
```

这个目标让 `force_feature` 不能塌缩成常量，因为常量 feature 无法解释不同样本的未来力均值、波动、峰值、趋势和强度。

## 2. Asymmetric distillation

student 仍然来自原来的 prefix semantic query：

```text
query_feature = normalize(force_semantic_query_proj(prefix_out[:, 0, :]))
```

teacher 来自物理锚定后的 force branch：

```text
force_feature = normalize(force_raw_feature)
```

distillation loss：

```text
L_distill = 1 - cosine(query_feature, stopgrad(force_feature))
```

关键是 `stopgrad(force_feature)`。这样 prefix semantic query 学习 force branch 中已经被物理 summary 约束住的接触动力学表示，但不会反向把 force branch 拖向常量方向。

旧配置字段 `force_target_loss_weight` 现在继续作为 distillation 权重使用，保留 CLI 和旧日志兼容。

## 总 loss

force-guided 配置现在包含：

```text
L = L_flow
  + force_loss_weight * L_force_L2
  + force_physical_loss_weight * L_phys
  + force_target_loss_weight * L_distill
```

当前 flexiv pump force-guided 配置默认：

```text
force_loss_weight = 0.05
force_physical_loss_weight = 0.05
force_target_loss_weight = 0.01
```

## 修改文件

```text
src/openpi/models/pi0_config.py
```

新增：

```text
force_physical_loss_weight
```

```text
src/openpi/models/pi0.py
```

新增或修改：

```text
_force_semantic_raw_feature
_force_target_summary
_force_physical_anchor_loss
_force_distillation_loss
force_prediction_trace
_compute_loss_and_info
```

```text
src/openpi/training/config.py
```

四个 flexiv pump force-guided 配置默认加入：

```text
force_physical_loss_weight=0.05
```

```text
scripts/train.py
scripts/analyze_force_metrics.py
```

加入新的日志和分析指标。

## 训练命令

原命令可以继续使用：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_lora_force_guided \
  --overwrite
```

这条命令对应 FRA 两阶段流程的阶段一。阶段二必须加载该阶段 checkpoint，并冻结本页描述的 TCN、
physical anchor 与 distillation 分支；完整命令见
[`flexiv_pump_pi0_finetune.md`](flexiv_pump_pi0_finetune.md)。

如果想单独调权重：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_lora_force_guided_cdt \
  --model.force-physical-loss-weight=0.05 \
  --model.force-target-loss-weight=0.01 \
  --overwrite
```

## 重点监控指标

新增指标：

```text
loss_force_physical_anchor
loss_force_physical_anchor_weighted
force_physical_summary_mse
force_physical_summary_mae
force_physical_summary_target_std_mean
force_physical_summary_pred_std_mean
loss_force_distill
loss_force_distill_weighted
force_distill_cosine_mean
```

兼容旧 dashboard 的指标仍会记录：

```text
loss_force_semantic_align
force_semantic_cosine_mean
```

但它们现在表示 prefix semantic query 到 stop-gradient force feature 的 distillation，而不是原来的 symmetric semantic alignment。

trace 文件里的：

```text
query_feature
```

仍然对应 prefix semantic query feature。因此新的 `query_batch_std_mean` 仍然是 semantic query 的 batch diversity。

## 判断是否起效

理想现象：

```text
force_physical_summary_mse 下降
loss_force_distill 下降
force_batch_std_mean 不再持续接近 0
query_batch_std_mean 不再持续接近 0
force prediction 与 target 更接近
主任务 loss_fm 不明显变坏
```

如果 `force_physical_summary_pred_std_mean` 也接近 0，而 target std 明显非 0，说明 physical anchor 仍没有学起来，可以提高：

```text
--model.force-physical-loss-weight=0.1
```

如果主任务 `loss_fm` 变差，优先降低：

```text
--model.force-target-loss-weight=0.005
```

## 论文表述建议

可以把这个模块描述为：

```text
Physics-anchored contact dynamics distillation.
```

核心叙事：

```text
Naive cross-modal cosine alignment admits a degenerate constant solution in low-diversity contact-rich data.
We therefore anchor the force encoder with future force statistics and distill the resulting contact-dynamics representation into a prefix semantic query using a stop-gradient teacher.
This keeps the module lightweight while making the learned representation physically identifiable.
```
