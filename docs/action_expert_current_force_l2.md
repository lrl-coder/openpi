# Action Expert：当前瞬时力线性条件与 L2 力预测

## 变更记录

- 分支：`action-expert-current-force-L2-loss`
- 记录日期：2026-07-24
- 影响范围：JAX `Pi0` 的 force-aware action expert、辅助力预测头 `F_phi`、训练诊断与 CoRACE 在线执行
- 不影响范围：未启用 `force_guidance` 的原始 π0、flow-matching 主损失、VLM semantic-force query、全局历史力 teacher

## 目标

本分支验证两个相互配套的简化：

1. action expert 不再编码 16 帧局部力历史，只接收控制时刻的当前 6D 力/力矩；
2. `F_phi` 不再学习异方差高斯分布，不再输出 `log_sigma`，改为直接回归未来力并使用 L2/MSE。

这样可以把 action expert 的力条件严格限制在当前观测，减少局部力编码器参数和历史窗口引入的时序混杂；同时消除 Gaussian NLL 中方差分支对优化目标的影响。

## 架构变化

### 变更前

```text
force_history_local [B, 16, 6]
    -> causal Conv1D
    -> LayerNorm + SiLU
    -> causal Conv1D
    -> LayerNorm + SiLU
    -> causal Conv1D
    -> temporal mean pooling
    -> Linear
    -> action-expert force token [B, 1, D]

action hidden h_t + Linear(action_hat_0)
    -> Linear(2 * force_dim)
    -> mu_f, log_sigma_f
    -> diagonal Gaussian NLL
```

### 变更后

```text
current force f_t [B, 6]
    -> Linear(6, D)
    -> action-expert force token [B, 1, D]

action hidden h_t + Linear(stopgrad(action_hat_0))
    -> Linear(force_dim)
    -> force_prediction [B, action_horizon, 6]
    -> L2 / MSE
```

其中：

```text
action_hat_0 = x_t - t * v_t
```

仍沿用上一版实现，并继续对 `action_hat_0` 使用 `stop_gradient`。因此辅助力损失会通过 `h_t` 更新 action expert 表征，并训练 `F_phi`，但不会经由 `action_hat_0` 分支直接改写 velocity head。

## 损失定义

对 batch 中第 `b` 个样本、action chunk 第 `k` 步：

```text
L_force[b, k]
    = (1 / force_dim)
      * sum_j (force_target[b, k, j] - force_prediction[b, k, j])^2
```

总损失保持原来的逐 action-step 形状：

```text
L_total = L_FM + force_loss_weight * L_force
```

默认 force-guided 配置仍使用：

```text
force_loss_weight = 0.05
```

训练指标从 `loss_force_nll` / `force_nll` 改为：

```text
loss_force_l2
force_l2
loss_force_weighted
force_residual_mse_mean
force_residual_rmse_mean
force_residual_mae_mean
force_residual_{mse,rmse,mae}_axis_0..5
```

同时记录预测值与标签的 batch 方差，便于识别 L2 回归退化为条件均值：

```text
force_prediction_batch_var_mean
force_true_var_mean
force_prediction_var_abs_gap_mean
```

## 当前力的选择规则

action expert 优先读取：

```text
observation.force
```

为了兼容已有数据和推理调用，如果该字段缺失，会依次回退到：

```text
force_history_local[:, -1]
force_history[:, -1]
force_history_global[:, -1]
zeros
```

这些回退都只取最后一个时刻，不会把历史序列送入 action expert。正常的 Flexiv force-guided 数据链路始终显式提供 `observation.force`。

## 全局历史力路径保持不变

本改动只删除 action expert 的局部历史力编码。以下训练路径仍保留：

```text
force_history_global [B, 64, 6]
    -> force semantic TCN teacher
    -> physical summary anchor
    -> semantic-force query distillation
```

因此，`force_history_global` 仍是训练 semantic-force teacher 所必需的输入。数据管线继续产生
`force_history_local` / `force_history`，仅用于数据兼容和诊断，不再参与 action token 或在线残差的构造。

## 在线执行：CoRACE

旧 token 缩放路径已删除。残差不再缩放 force token，也不进入 denoising。推理期由
**CoRACE（Contact-Residual Adaptive Chunk Execution）**把 `F_phi` 的预测作为 action chunk 的执行监测信号：

```text
policy(o_t) -> action_chunk[0:H], force_prediction[0:H]
execute action[k]
observe force[k]
compare force[k] with force_prediction[k]
if calibrated residual is exceeded:
    discard action[k+1:H]
    replan from the latest observation
```

在线残差使用独立尺度消除 6 个轴的单位与量级差异：

```text
r_k = mean_j(((force_observed[k, j] - force_prediction[k, j]) / force_scale[j]) ** 2)
```

`force_scale` 从训练划分的预测残差估计；阈值从独立 nominal calibration rollouts
用 chunk 最大残差的 split-conformal 分位数得到。完整协议、公式和部署方式见
[`contact_residual_adaptive_chunk_execution.md`](contact_residual_adaptive_chunk_execution.md)。

## 代码位置

- `src/openpi/models/pi0.py`
  - `force_current_proj`：当前力线性投影
  - `_embed_current_force_token`：构造单个当前力 token
  - `force_predictor_out`：直接输出 `force_dim`
  - `_force_l2_loss`：L2/MSE 辅助损失
  - `_force_prediction_info`：L2 诊断指标
- `src/openpi/policies/policy.py`
  - 可选返回与 action chunk 对齐的 `force_prediction`
  - 不再缓存上一轮残差或改变下一轮网络前向
- `packages/openpi-client/src/openpi_client/action_chunk_broker.py`
  - `CoRACEActionChunkBroker`：逐步对齐预测力与实测力
  - 超阈值时丢弃未执行 suffix 并从最新观测重规划
  - 提供残差尺度估计与 split-conformal 阈值标定函数
- `scripts/train.py`
  - 保存 `prediction`、`residual`、`squared_error` trace
- `scripts/analyze_force_metrics.py`
  - 分析 L2、残差和预测方差，不再分析 `log_sigma`
- `src/openpi/models/pi0_test.py`
  - 验证动作头只有一个当前力 Linear
  - 验证改变历史力不会改变当前力 token
  - 验证 `F_phi` 输出形状与 L2 数值

## 兼容性说明

### 原始 π0

`force_guidance=False` 时不会创建或调用新增力模块，原始动作训练和采样路径不变。

### 数据格式

现有 force-guided 数据格式不需要修改。当前力仍来自原始 13D state 的：

```text
force = observation.state[7:13]
```

### 旧 force-guided checkpoint

本分支对力头做了结构性修改：

- 删除 action expert 局部力卷积参数；
- 删除早期版本遗留但已不参与前向的 `force_predictor_in`；
- 新增 `force_current_proj`；
- `force_predictor_out` 的输出维度从 `2 * force_dim` 改为 `force_dim`。

因此不要从旧 NLL force-guided checkpoint 直接 `resume`。应从 π0 base 权重启动一个新的实验目录，让新力层随机初始化后训练。旧式 `(mu, log_sigma)` 推理输出和残差 token modulation 参数已不再兼容。

## 建议训练命令

LoRA force-guided：

```bash
uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_lora_current_force_l2 \
  --overwrite
```

建议使用全新的 `exp-name`，避免覆盖或误恢复旧 NLL 实验。

## 验收标准

1. `force_current_proj` 权重形状为 `[force_dim, action_expert_width]`；
2. action expert force token 对 `force_history_local/global` 的变化不敏感，只随 `force` 变化；
3. `force_predictor_out.out_features == force_dim`；
4. 日志包含 `loss_force_l2`，不再产生 `loss_force_nll` 或 `log_sigma` 指标；
5. `predict_force` 输出形状为 `[B, action_horizon, force_dim]`；
6. `return_force_prediction=True` 返回物理单位的 `[action_horizon, force_dim]` 预测；
7. CoRACE 只比较已执行动作的同索引预测，超阈值时立即重规划；
8. baseline π0、force-aware 模型与 CoRACE 单测全部通过。

## 2026-07-24 验证结果

静态检查：

```text
ruff check（全部变更 Python 文件）: passed
git diff --check: passed
```

自动测试：

```text
pytest -m "not manual" \
  packages/openpi-client/src/openpi_client/action_chunk_broker_test.py \
  src/openpi/models/pi0_test.py \
  src/openpi/policies/policy_test.py \
  src/openpi/transforms_test.py

27 passed, 2 deselected

pytest packages/openpi-client/src/openpi_client
28 passed
```

实际 JAX 前向：

```text
force-guided dummy model:
  compute_loss_info -> loss.shape = (1, 4)
  sample_actions     -> actions.shape = (1, 4, 8)
  predict_force     -> prediction.shape = (1, 4, 6)
  loss_force_l2     -> finite
  all outputs finite
```

CoRACE：

```text
strict action/force index alignment: passed
sensor-delay alignment: passed
residual trigger and suffix discard: passed
split-conformal threshold calibration: passed
physical-unit force output transform: passed
```
