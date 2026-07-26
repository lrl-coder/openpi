# Action Expert：当前瞬时力条件与 Concat-MLP 绝对力预测

## 变更记录

- 分支：`action-expert-current-force-L2-loss`
- 初始记录：2026-07-24，当前瞬时力条件与 L2 点预测
- 最新更新：2026-07-26，`F_phi` 改为 `concat + 两层 MLP`
- 影响范围：JAX `Pi0` 的 force-aware action expert、训练期辅助力预测头 `F_phi` 与训练诊断
- 不影响范围：未启用 `force_guidance` 的原始 π0、flow-matching 主损失、VLM semantic-force query、全局历史力 teacher

## 目标

本分支验证三个相互配套的设计：

1. action expert 不再编码 16 帧局部力历史，只接收控制时刻的当前 6D 力/力矩；
2. `F_phi` 不再学习异方差高斯分布，不再输出 `log_sigma`，改为直接回归未来力并使用 L2/MSE。
3. `F_phi` 不再把 action hidden 与 clean-action embedding 直接相加，而是保留两者的独立语义，
   拼接后用一个小型两层 MLP 学习非线性的状态—动作融合。

这样可以把 action expert 的力条件严格限制在当前观测，减少局部力编码器参数和历史窗口引入的时序混杂；同时消除 Gaussian NLL 中方差分支对优化目标的影响。
Concat-MLP 则避免预设 action hidden 和动作向量必须在同一个 latent space 中逐元素对齐，使力预测头能够显式学习
“相同动作在不同场景和接触状态下产生不同未来力”的条件关系。

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

### 当前版本

```text
current force f_t [B, 6]
    -> Linear(6, D)
    -> action-expert force token [B, 1, D]

action hidden h_tau [B, action_horizon, D]
    -> LayerNorm
                                      \
                                       -> concat [B, action_horizon, D + action_dim]
                                      /
stopgrad(action_hat_0) [B, action_horizon, action_dim]
    -> Linear(D + action_dim, M)
    -> SiLU
    -> Linear(M, force_dim)
    -> absolute force_prediction [B, action_horizon, 6]
    -> L2 / MSE
```

其中：

```text
action_hat_0 = x_tau - tau * v_tau
D = action expert hidden width
M = force_predictor_hidden_dim，默认 256
```

这里用 `tau` 表示 flow-matching 的加噪时间，避免和真实控制时刻 `t` 混淆。

`action_hat_0` 继续使用 `stop_gradient`。因此辅助力损失会通过 `h_tau` 更新 action expert 表征，
并训练 `F_phi` 的 LayerNorm 与两层 MLP，但不会经由
`action_hat_0 -> v_tau -> action_out_proj` 分支直接改写 velocity head。

显式的 clean-action 分支不再复用主动作输入层 `action_in_proj`。`action_hat_0` 以动作空间中的原始
归一化向量直接参与 concat，随后由 `force_predictor_fusion_in` 学习专用于未来力预测的映射。

### 为什么使用 concat + 两层 MLP

#### 论文撰写时的核心动机

`h_tau` 和 `action_hat_0` 提供的是两类互补信息：

- `h_tau`：场景、当前力、机器人状态、语言任务以及去噪过程的综合表征；
- `action_hat_0`：模型当前认为最终会执行的 clean action。

未来力既不是只由环境决定，也不是只由动作决定，而是二者共同作用的结果：

```text
未来力 = 条件动力学（环境状态，执行动作）
```

因此，`F_phi` 应同时接收：

```text
环境与接触上下文 h_tau
    +
模型将要执行的动作 action_hat_0
    ->
未来绝对力 force_prediction
```

从条件动力学的角度，可以写成：

```text
force_prediction = F_phi(h_tau, stopgrad(action_hat_0))
```

其中 `h_tau` 回答“机器人当前处于什么场景和接触状态”，`action_hat_0` 回答“机器人准备执行什么动作”。
将二者融合后预测未来力，使辅助任务约束 action expert 学习环境—动作—接触后果之间的关系，而不只是拟合
当前力的时间相关性或从动作单独回归力。

对 action chunk 第 `k` 步，当前实现等价于：

```text
h_norm[b, k] = LayerNorm(h_tau[b, k])
z[b, k] = concat(h_norm[b, k], stopgrad(action_hat_0[b, k]))
hidden[b, k] = SiLU(W_in z[b, k] + b_in)
force_prediction[b, k] = W_out hidden[b, k] + b_out
```

两类输入的含义不同：

- `h_tau` 包含图像、语言、机器人状态、当前力和 denoising context；
- `action_hat_0` 明确表示模型当前估计的最终 clean action。

直接相加会预设二者在相同 latent 坐标中逐元素对齐，而且其后的单个 Linear 只能把 clean action
作为线性修正。Concat 保留两类信息，SiLU 则允许 MLP 学习非线性状态—动作交互。
瓶颈宽度 `M=256` 相对于 action expert 很小，因此 `F_phi` 仍是轻量辅助头，而不是另一个大型动力学模型。

### 仍然预测绝对力

本改动不改变监督目标。`F_phi` 直接输出未来绝对力/力矩：

```text
force_prediction[b, k] ~= force_target[b, k]
```

不会构造：

```text
force_delta = force_target - current_force
```

也不会在输出端执行：

```text
force_prediction = current_force + predicted_delta
```

当前力只作为 action expert 的观测条件；`F_phi` 的标签和输出始终是 action horizon 对齐的绝对 6D 力。

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
`force_history_local` / `force_history`，仅用于数据兼容和诊断，不参与 FRA。

## 在线执行：FRA

`F_phi` 保留为阶段一训练期辅助目标和离线诊断，不再返回给机器人，也不再构造在线预测力残差。

推理期由 [FRA](force_reflex_adapter.md) 直接复用物理锚定的全局历史力 TCN：

```text
force_history_global -> Contact Dynamics Token
latest q_t + nominal action + chunk progress
    -> Force Reflex Adapter
    -> joint residual
    -> corrected joint target
```

因此在线路径不再需要 per-axis residual scale、conformal threshold、预测/测量索引对齐或触发式丢弃
chunk suffix。

## 代码位置

- `src/openpi/models/pi0.py`
  - `force_current_proj`：当前力线性投影
  - `_embed_current_force_token`：构造单个当前力 token
  - `force_predictor_hidden_norm`：归一化 action hidden
  - `force_predictor_fusion_in`：将 concat 后的 `D + action_dim` 特征映射到小型 MLP 隐层
  - `force_predictor_out`：将 MLP 隐层直接映射为绝对 `force_dim`
  - `_decode_force_from_action_features`：执行 concat、SiLU 和绝对力 readout
  - `_force_l2_loss`：L2/MSE 辅助损失
  - `_force_prediction_info`：L2 诊断指标
- `src/openpi/models/pi0_config.py`
  - `force_predictor_hidden_dim`：两层 MLP 的瓶颈宽度，默认 `256`
- `src/openpi/policies/policy.py`
  - `fra_mode=True` 时只运行 Contact Dynamics Encoder 与 FRA
  - 正常请求仍生成 nominal action chunk
- `packages/openpi-client/src/openpi_client/action_chunk_broker.py`
  - `ForceReflexActionChunkBroker`：每步更新力历史并连续修正当前关节目标
  - 保持夹爪 nominal command，并在下发前施加机械臂关节限位
- `scripts/train.py`
  - 保存 `prediction`、`residual`、`squared_error` trace
- `scripts/analyze_force_metrics.py`
  - 分析 L2、残差和预测方差，不再分析 `log_sigma`
- `src/openpi/models/pi0_test.py`
  - 验证动作头只有一个当前力 Linear
  - 验证改变历史力不会改变当前力 token
  - 验证 `F_phi` 使用 `D + action_dim -> M -> force_dim` 的两层 MLP
  - 验证绝对力输出形状与 L2 数值

## 兼容性说明

### 原始 π0

`force_guidance=False` 时不会创建或调用新增力模块，原始动作训练和采样路径不变。

### 数据格式

当前 joint-space 数据是 14D state。当前力来自：

```text
force = observation.state[8:14]
```

### 旧 force-guided checkpoint

本分支对力头做了结构性修改：

- 删除 action expert 局部力卷积参数；
- 删除早期版本遗留但已不参与前向的 `force_predictor_in`；
- 新增 `force_current_proj`；
- 新增 `force_predictor_hidden_norm` 与 `force_predictor_fusion_in`；
- `force_predictor_out` 的输入维度由 action expert width 改为 `force_predictor_hidden_dim`；
- `force_predictor_out` 仍只输出绝对 `force_dim`，不输出力增量或 `log_sigma`。

因此不要从旧 NLL force-guided checkpoint，或 2026-07-24 的 Add-Linear L2 checkpoint 直接 `resume`。
应从 π0 base 权重启动一个新的实验目录，让新力层随机初始化后训练。
旧式 `(mu, log_sigma)` 输出和 Add-Linear 力头参数都不再兼容。

## 建议训练命令

LoRA force-guided：

```bash
uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_lora_current_force_concat_mlp_l2 \
  --overwrite
```

建议使用全新的 `exp-name`，避免覆盖或误恢复旧 NLL 或 Add-Linear 实验。

## 验收标准

1. `force_current_proj` 权重形状为 `[force_dim, action_expert_width]`；
2. action expert force token 对 `force_history_local/global` 的变化不敏感，只随 `force` 变化；
3. `force_predictor_fusion_in.in_features == action_expert_width + action_dim`；
4. `force_predictor_fusion_in.out_features == force_predictor_hidden_dim`；
5. `force_predictor_out` 的输入/输出维度为 `force_predictor_hidden_dim -> force_dim`；
6. `F_phi` 不复用 `action_in_proj`，且不把 `current_force` 加回预测；
7. 日志包含 `loss_force_l2`，不再产生 `loss_force_nll` 或 `log_sigma` 指标；
8. `predict_force` 仅作为训练/诊断 API 输出绝对力 `[B, action_horizon, force_dim]`；
9. 在线服务不再接受 `return_force_prediction`；
10. FRA reflex-only 请求不调用完整 action sampler；
11. 夹爪命令不被 FRA 修改。
