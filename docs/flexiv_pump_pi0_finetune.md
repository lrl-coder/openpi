# Flexiv Pump：force-guided π0 与 FRA 两阶段微调

本文给出当前 joint-space 数据格式、两阶段训练命令和 FRA 部署入口。完整算法、公式与消融见
[`force_reflex_adapter.md`](force_reflex_adapter.md)。

## 数据格式

原始观测：

```text
observation/state =
  joint_position(7), gripper_width(1), f_ext_base_frame(6)  # 14D
```

原始动作：

```text
action =
  target_joint_position(7), target_gripper_width(1)         # 8D
```

FRA 只修正前 7 个机械臂关节；第 8 维夹爪命令始终由 π0 控制。模型内部仍把 state/action pad 到
`action_dim=32`，输出时裁回 8D。

所有 `*_force_guided` 和 `*_fra_stage2` 配置要求：

```text
force                 = state[8:14]
force_history_global  = 64-step causal history
force_targets         = action horizon 对齐的未来力
```

阶段二还构造：

```text
fra_joint_states  [H, 7]
fra_force_history [H, 64, 6]
```

第 `k` 项分别是当前状态 `q_(tau+k)` 与结束于 `tau+k` 的 causal force history，供 stale-chunk 随机采样。

## 环境变量

```bash
cd /root/autodl-tmp/openpi

export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-fs/openpi_cache/hf_datasets
export HF_HOME=/root/autodl-fs/openpi_cache/hf_home
```

## 归一化统计

先为完整 14D state 与 8D action 计算统计量：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce_lora_with_force
```

旧 end-effector 数据生成的 13D state stats 与当前 joint-space 配置不兼容；force-guided 配置会直接拒绝
加载它们。切换数据后必须重新运行上面的命令并确认生成的 `state.mean` 长度为 14、`actions.mean` 长度为 8。

force-guided 配置会从 `state[8:14]` 派生 force stats，并为 FRA 增加以下别名：

```text
fra_force_history           -> force stats
fra_joint_states            -> action stats
fra_current_joint_state     -> action stats
fra_nominal_action          -> action stats
fra_previous_nominal_action -> action stats
```

让 `q_t` 使用 action stats 是必要的：这样 `nominal_action - q_t` 在归一化空间仍是有意义的关节误差。

## 阶段一：force-guided π0

推荐配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora_force_guided
```

目标：

```text
L_base = L_flow
       + 0.05 * L_force_L2
       + 0.05 * L_physical_anchor
       + 0.01 * L_contact_distill
```

训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_fra_stage1 \
  --overwrite
```

阶段一保留：

- π0 flow-matching action chunk；
- 当前瞬时力 action token；
- Contact Dynamics TCN；
- future-force physical summary anchor；
- prefix query asymmetric distillation；
- 训练期 `F_phi` L2 辅助力预测。

`F_phi` 只用于阶段一辅助监督和离线诊断，不再提供在线 force residual。

阶段一重点指标：

```text
loss_fm
loss_force_l2
loss_force_physical_anchor
loss_force_distill
force_physical_summary_mse
force_physical_summary_pred_std_mean
force_distill_cosine_mean
```

## 阶段二：冻结主模型，只训练 FRA

推荐配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora_fra_stage2
```

选定阶段一 checkpoint 的 `params` 目录：

```bash
STAGE1_PARAMS=/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/flexiv_pump_fra_stage1/19999/params
```

启动全新的阶段二实验：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_fra_stage2 \
  --exp-name=flexiv_pump_fra_stage2 \
  --weight-loader.params-path="$STAGE1_PARAMS" \
  --overwrite
```

阶段二的实际训练步骤：

1. 构建带 FRA 的同结构模型；
2. 加载阶段一完整 checkpoint；
3. 将阶段一不存在的 `force_reflex_adapter` 参数做零残差初始化；
4. 冻结 π0、LoRA、action expert、Contact Dynamics Token 及所有辅助头；
5. 由冻结 π0 对旧观测 \(o_\tau\) 生成 nominal chunk；
6. 对每个样本随机取 `chunk_age = ell`；
7. 用 `q_(tau+ell)`、结束于该时刻的 force history 和 nominal chunk 第 `ell` 项运行 FRA；
8. 用同一时刻专家动作做 Huber 监督，并添加 residual magnitude 与 smoothness 正则；
9. optimizer 只更新路径包含 `force_reflex_adapter` 的参数。

阶段二默认：

```text
steps        = 10_000
warmup       = 500
peak LR      = 1e-4
final LR     = 1e-5
batch size   = 16
EMA          = off
```

不要把阶段一目录直接 `--resume` 成阶段二。两者的模型参数树和 optimizer state 不同。

阶段二监控：

```text
loss_fra_huber
loss_fra_magnitude
loss_fra_smoothness
fra_chunk_age_mean
fra_residual_abs_mean
fra_gate_mean
fra_corrected_action_mae
```

全量微调对应：

```text
阶段一：pi0_flexiv_pump_1bottle_inputForce_force_guided
阶段二：pi0_flexiv_pump_1bottle_inputForce_fra_stage2
```

## 部署

服务阶段二 checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora_fra_stage2 \
  --policy.dir=/path/to/fra_stage2/checkpoint
```

机器人端使用：

```python
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
policy = action_chunk_broker.ForceReflexActionChunkBroker(
    client,
    action_horizon=50,
    arm_joint_dim=7,
    force_dim=6,
    force_history_len=64,
    joint_lower_limits=ARM_JOINT_LOWER_LIMITS,
    joint_upper_limits=ARM_JOINT_UPPER_LIMITS,
)
```

每次 `policy.infer(observation)` 返回一个已修正并施加关节限位的 8D 动作。chunk 为空时才运行完整 π0；
其余控制周期只运行 Contact Dynamics Encoder 与 FRA。

输入至少包含：

```python
observation = {
    "observation/image": base_image,
    "observation/wrist_image": wrist_image,
    "observation/state": np.concatenate([q_t, [gripper_width], wrench_t]),
    "force": wrench_t,
    "prompt": task_instruction,
}
```

## 最小验收

1. 原始 state 是 14D，action 是 8D，关节顺序与机器人控制器一致；
2. action、state、force 时间戳逐控制步对齐；
3. 阶段一 checkpoint 的 physical anchor 和 distillation 均已收敛；
4. 阶段二日志确认只有 FRA 参数产生梯度；
5. `fra_chunk_age_mean` 符合均匀采样，residual 不长期严格为零；
6. reflex-only 请求不调用 π0 action sampler；
7. 输出夹爪维与 nominal chunk 完全一致；
8. 下发前执行厂商 joint limits，急停和力硬限位独立于 FRA；
9. 对比 Fixed Chunk、Step-wise Replanning、旧 CoRACE baseline 和 FRA 时使用相同阶段一主策略。
