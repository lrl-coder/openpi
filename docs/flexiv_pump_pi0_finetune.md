# Flexiv Pump：π0 微调与 CoRACE 部署

本文只描述当前实现。旧 Gaussian-NLL、采样能量引导和残差 token modulation 路径已经删除。

## 环境变量

每次训练前先进入 openpi 目录并设置环境变量：

```bash
cd /root/autodl-tmp/openpi

export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-fs/openpi_cache/hf_datasets
export HF_HOME=/root/autodl-fs/openpi_cache/hf_home
```

## 归一化统计

训练前必须先有 norm stats。

这些 stats 来自完整 13 维 state。默认 7 维配置归一化时只会取前 7 维统计量。如果想重新生成更干净的 7 维统计，或者文件不存在，运行：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce_lora
```

如果要训练带力信息的 LoRA 配置，需要单独计算：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce_lora_with_force
```

如果要跑全量微调配置，也需要给全量配置计算一次：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce
```

带力信息的全量配置对应：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce_with_force
```

## 配置选择

推荐主实验：

```text
pi0_flexiv_pump_1bottle_inputForce_lora_force_guided
```

对照配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora
pi0_flexiv_pump_1bottle_inputForce_lora_with_force
pi0_flexiv_pump_1bottle_inputForce_lora_with_force_guided
pi0_flexiv_pump_1bottle_inputForce
pi0_flexiv_pump_1bottle_inputForce_with_force
pi0_flexiv_pump_1bottle_inputForce_force_guided
pi0_flexiv_pump_1bottle_inputForce_with_force_guided
```

命名含义：

- `lora`：只训练 LoRA 与新增模块；
- `with_force`：常规 state token 也接收完整 13D state；
- `force_guided`：历史力语义 teacher、当前力 action token 和 `F_phi` 力预测头；名称为配置兼容保留，
  当前在线执行模块称为 CoRACE。

## 数据格式

原始观测 state：

```text
current_eef_pose(6), gripper_width(1), f_ext_base_frame(6)
```

原始动作：

```text
target_eef_pose(6), target_gripper_width(1)
```

所有 `*_force_guided` 配置都要求原始 13D state。默认主配置只把前 7 维作为常规 state token，
但单独提取：

```text
force                 = state[7:13]
force_history_global  = 64-step history
force_targets         = future force aligned to the action horizon
```

action expert 的力条件只有当前 `force` 经一个 Linear 投影；64 步全局历史只用于训练期 contact-dynamics
teacher。详细结构见
[`action_expert_current_force_l2.md`](action_expert_current_force_l2.md) 和
[`contact_dynamics_token_force_anchor.md`](contact_dynamics_token_force_anchor.md)。

## 当前训练目标

```text
L = L_flow
  + 0.05 * L_force_L2
  + 0.05 * L_physical_anchor
  + 0.01 * L_contact_distill
```

其中 `F_phi` 直接回归：

```text
force_prediction: [batch, action_horizon, 6]
```

不再输出 `log_sigma`，不再优化 Gaussian NLL。

## 训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_lora_current_force_l2 \
  --overwrite
```

旧 force-aware checkpoint 的力头结构不同，不能直接 resume。应从 π0 base 权重启动新实验目录。

重点监控：

```text
loss_fm
loss_force_l2
loss_force_weighted
force_residual_mse_mean
force_residual_rmse_mean
force_prediction_batch_var_mean
force_true_var_mean
loss_force_physical_anchor
loss_force_distill
force_distill_cosine_mean
```

若 `force_prediction_batch_var_mean` 长期接近 0，而 `force_true_var_mean` 明显非 0，说明 L2 头可能退化为
条件均值；此时先检查 action/force 时间对齐与数据方差，再调 loss weight。

## 固定 chunk 推理

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --policy.dir=/path/to/checkpoint
```

输入仍需包含原始 13D `observation/state`。

## CoRACE 闭环执行

服务端额外返回物理单位的未来力预测：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --policy.dir=/path/to/checkpoint \
  --policy.return-force-prediction
```

机器人端用 `CoRACEActionChunkBroker` 包装 websocket client。每个控制步都把最新 6D force 放在 observation
的 `force` 键中并调用 broker；broker 只在 chunk 耗尽或残差超阈值时访问服务端。

阈值必须从独立 nominal calibration rollouts 标定，不要复用训练 loss 或手写常数。完整公式、客户端代码、
传感器延迟设置和消融设计见
[`contact_residual_adaptive_chunk_execution.md`](contact_residual_adaptive_chunk_execution.md)。

## 最小验收

1. 数据中的 action 与 future force target 逐控制步对齐；
2. `predict_force` 输出 `[B, H, 6]` 且物理单位反归一化正确；
3. 离线按 horizon step 报告六轴 RMSE，而不只报总平均；
4. 固定 chunk、每步重规划、CoRACE 使用相同 checkpoint；
5. 同时报告成功率、峰值力、query 次数、平均实际 chunk 长度和误触发率；
6. 机器人安全限位与急停独立于 CoRACE。
