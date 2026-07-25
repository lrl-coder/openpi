# FRA：Force Reflex Adapter

本文是当前 FRA 设计与实现的唯一说明。FRA 已替换原来的 CoRACE 在线模块。
设计依据为用户提供的 [“力反馈闭环重构”共享对话](https://chatgpt.com/share/6a644c59-0880-83ea-a3c2-20feacdfbb75)；
本文把其中的结构、损失、两阶段训练和部署要求整理为与仓库代码一致的可执行版本。

一句话概括：

> **Plan in chunks, correct in joints.**
>
> π0 低频生成名义关节动作 chunk；FRA 根据每个控制周期的新力反馈，高频、连续地修正即将执行的关节目标。

FRA 不是异常检测器，也不输出 `continue/replan`。它直接输出关节空间动作残差，因此力反馈真正参与动作生成。

## 1. 为什么替换 CoRACE

CoRACE 属于 detect-and-replan：

```text
预测未来力
  -> 与实测力比较
  -> 超过标定阈值
  -> 截断 chunk
  -> 重新运行 π0
```

它依赖未来力预测、各轴残差尺度、传感器延迟对齐和 held-out calibration threshold，且把连续接触反馈压缩为“继续/重规划”二值事件。触发后仍需等待大模型生成新 chunk。

FRA 改成双时间尺度闭环：

```text
低频：image + language + q_tau -> π0 -> nominal joint-action chunk
高频：latest q_t + force history + nominal action -> FRA -> joint residual
```

## 2. 数学定义

时刻 \(\tau\)，π0 生成长度为 \(H\) 的名义关节动作：

\[
\bar{\mathbf A}_{\tau}
=
[\bar{\mathbf a}_{\tau,0},
 \bar{\mathbf a}_{\tau,1},
 \ldots,
 \bar{\mathbf a}_{\tau,H-1}],
\qquad
\bar{\mathbf a}_{\tau,k}\in\mathbb R^{N_a}.
\]

执行到 chunk 第 \(k\) 步时，FRA 读取：

- 当前机械臂关节角 \(\mathbf q_t\)；
- 最近 \(L\) 步 6D 力/力矩历史 \(\mathbf f_{t-L+1:t}\)；
- 当前名义动作 \(\bar{\mathbf a}_{\tau,k}\)；
- 前一个名义动作 \(\bar{\mathbf a}_{\tau,k-1}\)；
- chunk 进度 \(k/H\)。

物理锚定的 Contact Dynamics Encoder 给出：

\[
\mathbf c_t
=
E_f(\mathbf f_{t-L+1:t}).
\]

FRA 的输入与输出为：

\[
\delta\mathbf q_t
=
\mathrm{FRA}_{\psi}
\left(
\mathbf c_t,\,
\mathbf q_t,\,
\bar{\mathbf q}_{\tau,k}-\mathbf q_t,\,
\bar{\mathbf q}_{\tau,k}-\bar{\mathbf q}_{\tau,k-1},\,
\frac{k}{H}
\right).
\]

只修正机械臂关节，夹爪仍由 π0 控制：

\[
\mathbf a_t
=
\left[
\bar{\mathbf q}_{\tau,k}+\delta\mathbf q_t;\,
\bar g_{\tau,k}
\right].
\]

发给机器人前必须再施加厂商关节限位：

\[
\mathbf a_t^{\mathrm{cmd}}
=
\operatorname{clip}
\left(
\mathbf a_t,\,
\mathbf a_{\min},\,
\mathbf a_{\max}
\right).
\]

闭环因果链是：

```text
实际接触 -> 实测力 -> FRA 关节修正 -> 新的实际接触
```

## 3. 复用 Contact Dynamics Token

FRA 不增加第二套力编码器，直接复用阶段一训练并物理锚定的：

```text
force_history_global [L, 6]
  -> causal TCN
  -> temporal mean pooling
  -> force_semantic_out
  -> L2-normalized Contact Dynamics Token
```

完整结构：

```text
image, language, q_tau
          |
          v
π0 -> nominal action chunk -------------------+
                                              |
latest force history -> Contact Dynamics Token|
                                              v
latest q_t ---------------------------------> FRA -> delta q_t
                                              |
                                              v
                            nominal action + delta q_t
```

两个模块形成明确分工：

1. Contact Dynamics Token 提取物理可辨识的接触状态；
2. Force Reflex Adapter 把接触状态转换为关节空间动作修正。

## 4. FRA 网络

实现位于 `src/openpi/models/pi0.py`，网络是两层 MLP：

```python
contact_feature = contact_dynamics_encoder(force_history)

adapter_input = concat(
    contact_feature,
    current_joint_state,
    nominal_action - current_joint_state,
    nominal_action - previous_nominal_action,
    chunk_progress,
)

hidden = swish(input_proj(adapter_input))
hidden = swish(hidden_proj(hidden))
gate = sigmoid(gate_head(hidden))
direction = tanh(residual_head(hidden))
residual = gate * action_scale * direction
```

`residual_head` 零初始化，因此新挂接 FRA 时严格等于 identity correction。

当前默认：

```text
arm_joint_dim = 7
hidden_dim = 256
FRA 参数量 < 200k
```

FRA 在归一化动作空间中使用 `action_scale=1`。动作经标准差归一化，因此反归一化到物理空间后，每一维残差的自然尺度正好是该维动作标准差 \(\boldsymbol\sigma_a\)：

\[
\delta\mathbf q_t
=
\mathbf g_t
\odot
\boldsymbol\sigma_a
\odot
\tanh(\mathbf r_t).
\]

这复用训练数据已有的动作统计量，不增加残差尺度或阈值标定。

## 5. Stale-Chunk Residual Supervision

只增加 residual head 容易得到全零解。FRA 的关键不是 head，而是 stale-chunk 监督。

对旧计划生成时刻 \(\tau\)，随机采样 chunk age：

\[
\ell
\sim
\mathcal U\{0,\ldots,H_{\mathrm{train}}-1\},
\qquad
t=\tau+\ell.
\]

冻结 π0，并由旧观测生成名义 chunk：

\[
\bar{\mathbf A}_{\tau}
=
\operatorname{sg}\!\left(\pi_\theta(\mathbf o_\tau)\right).
\]

取旧 chunk 中对应当前时刻的动作：

\[
\bar{\mathbf a}_{\tau,\ell}.
\]

数据集中同一时刻的专家动作是 \(\mathbf a_t^*\)。隐式残差目标为：

\[
\delta\mathbf a_t^*
=
\mathbf a_t^*
-
\operatorname{sg}\!\left(\bar{\mathbf a}_{\tau,\ell}\right).
\]

实现不显式回归 residual，而是监督修正后的动作：

\[
\mathcal L_{\mathrm{FRA}}
=
\operatorname{Huber}
\left(
\bar{\mathbf a}_{\tau,\ell}+\delta\mathbf a_t,\,
\mathbf a_t^*
\right).
\]

正则项：

\[
\mathcal L_{\mathrm{reg}}
=
\lambda_{\mathrm{mag}}\|\delta\mathbf a_t\|_1
+
\lambda_{\mathrm{smooth}}
\|\delta\mathbf a_t-\delta\mathbf a_{t-1}\|_2^2.
\]

当前默认：

```text
lambda_FRA    = 1.0
lambda_mag    = 1e-3
lambda_smooth = 1e-2
Huber delta   = 1.0
max chunk age = 49
```

训练样本以 \(\tau\) 时刻图像、语言、关节状态生成旧计划，同时从 timestamp-expanded trajectory 中取：

```text
q_tau ... q_(tau+H-1)
force_history ending at tau ... tau+H-1
expert action a*_tau ... a*_(tau+H-1)
```

因此训练条件与部署一致：旧 chunk 来自 \(\tau\)，反馈来自当前 \(t=\tau+\ell\)。

不需要人工干预、失败恢复、额外 correction trajectory、接触标签、IK/FK、残差尺度或 calibration split。

## 6. 必须采用的两阶段训练

默认论文与复现实验只采用两阶段训练。不要一开始联合训练 π0 和 FRA，否则主策略可能把动作误差转移给 residual head。

### 6.1 数据和归一化前提

当前 Flexiv joint-space 数据约定：

```text
observation/state:
  joint_position(7), gripper_width(1), f_ext_base_frame(6)  # 共 14 维

action:
  target_joint_position(7), target_gripper_width(1)         # 共 8 维
```

先设置环境：

```bash
cd /root/autodl-tmp/openpi

export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-fs/openpi_cache/hf_datasets
export HF_HOME=/root/autodl-fs/openpi_cache/hf_home
```

计算一次包含完整 14D state 与 8D action 的统计量：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce_lora_with_force
```

阶段一和阶段二必须复用同一份 norm stats。不要在阶段二重新统计。

### 6.2 阶段一：训练 force-guided π0

阶段一目标：

\[
\mathcal L_{\mathrm{base}}
=
\mathcal L_{\mathrm{flow}}
+
\lambda_{\mathrm{force}}\mathcal L_{\mathrm{force}}
+
\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}}
+
\lambda_{\mathrm{distill}}\mathcal L_{\mathrm{distill}}.
\]

默认权重：

```text
lambda_force   = 0.05
lambda_phys    = 0.05
lambda_distill = 0.01
```

LoRA 推荐命令：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_fra_stage1 \
  --overwrite
```

阶段一需要确认：

```text
loss_fm                         持续下降
loss_force_l2                   有限并下降
loss_force_physical_anchor      有限并下降
loss_force_distill              有限并下降
force_physical_summary_pred_std_mean 不长期塌缩到 0
```

选定最终阶段一 checkpoint，例如：

```text
/root/autodl-fs/openpi_checkpoints/
  pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/
  flexiv_pump_fra_stage1/
  19999/
  params
```

### 6.3 阶段二：冻结 π0 和 Contact Dynamics Token，只训练 FRA

阶段二配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora_fra_stage2
```

它会：

1. 加载阶段一完整 checkpoint；
2. 为缺失的 FRA 参数做 identity 初始化；
3. 冻结 PaliGemma、action expert、action head、force predictor、Contact Dynamics TCN、physical anchor 和 distillation head；
4. 每个 batch 用冻结 π0 生成 nominal chunk；
5. 随机采样 \(\ell\)，只优化 FRA Huber、幅值和平滑损失。

命令：

```bash
STAGE1_PARAMS=/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/flexiv_pump_fra_stage1/19999/params

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_fra_stage2 \
  --exp-name=flexiv_pump_fra_stage2 \
  --weight-loader.params-path="$STAGE1_PARAMS" \
  --overwrite
```

注意：

- `STAGE1_PARAMS` 必须指向 checkpoint 的 `params` 子目录；
- 阶段二是新实验，不能对阶段一使用 `--resume`；
- 阶段二默认 10k steps、peak LR `1e-4`、500-step warmup；
- 阶段二 checkpoint 同时保存冻结主模型和 FRA，可直接部署；
- 全量微调对应配置为 `pi0_flexiv_pump_1bottle_inputForce_fra_stage2`。

阶段二重点监控：

```text
loss_fra_huber
loss_fra_huber_weighted
loss_fra_magnitude
loss_fra_smoothness
fra_chunk_age_mean
fra_residual_abs_mean
fra_gate_mean
fra_corrected_action_mae
grad_norm
```

`fra_chunk_age_mean` 应接近采样范围中点。`fra_residual_abs_mean` 若长期严格为 0，先检查阶段一 checkpoint 是否正确加载、trajectory 时间戳是否展开，以及 action/state 是否都是相同关节顺序和单位。

## 7. 推理和机器人端执行

服务阶段二 checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora_fra_stage2 \
  --policy.dir=/path/to/fra_stage2/checkpoint
```

机器人端：

```python
import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

policy = action_chunk_broker.ForceReflexActionChunkBroker(
    client,
    action_horizon=50,
    arm_joint_dim=7,
    force_dim=6,
    force_history_len=64,
    joint_lower_limits=np.asarray(ARM_JOINT_LOWER_LIMITS),
    joint_upper_limits=np.asarray(ARM_JOINT_UPPER_LIMITS),
)

for _ in range(num_control_steps):
    observation = {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate([q_t, [gripper_width], wrench_t]),
        "force": wrench_t,
        "prompt": task_instruction,
    }
    result = policy.infer(observation)
    robot.execute_joint_target(result["actions"])
    log(result["fra"])
```

执行语义：

```text
chunk 为空：
  运行一次 π0，得到 nominal chunk

每个控制周期：
  更新 64-step force history
  读取 q_t
  只运行 Contact Dynamics Encoder + FRA
  修正 nominal_chunk[k] 的前 7 个机械臂关节
  保留 nominal gripper
  施加 joint limits
  下发 corrected action
```

参考 broker 每个控制周期通过同一服务请求 reflex-only 前向，但不会重新运行 π0。若部署环境需要消除逐步网络往返，可把冻结的 Contact Dynamics Encoder 与 FRA 一并导出到机器人 client；算法与输入输出不变。
FRA 的目标运行频率是机器人低层控制允许的 15–50 Hz 或更高频率，π0 仍只按 chunk 周期低频运行。

`result["fra"]` 包含：

```text
chunk_index
chunk_progress
plan_count
nominal_action
corrected_action
residual
gate
```

## 8. 关节角表示与无 IK/FK

FRA 完全在关节空间工作。

绝对关节目标：

\[
\mathbf q_{t+1}^{\mathrm{cmd}}
=
\bar{\mathbf q}_{t+1}
+
\delta\mathbf q_t.
\]

若另一数据集使用关节增量，也可在同一表示中做：

\[
\Delta\mathbf q_t^{\mathrm{cmd}}
=
\overline{\Delta\mathbf q}_t
+
\delta\Delta\mathbf q_t.
\]

不需要：

\[
\mathbf x=\mathrm{FK}(\mathbf q),
\qquad
\mathbf q=\mathrm{IK}(\mathbf x).
\]

必须保证：

- state、action 与机器人控制器使用同一关节顺序；
- 角度单位一致；
- observation、force 与 action 时间戳同步；
- 低层控制器接受 joint position target；
- 厂商关节限位、速度限制、急停独立存在。

## 9. FRA 与旧 CoRACE 的区别

| 方面 | CoRACE | FRA |
| --- | --- | --- |
| 力反馈用途 | 判断 chunk 是否失效 | 直接修正下一步动作 |
| 闭环 | 事件触发、二值 | 连续、高频 |
| 输出 | continue / replan | \(\delta\mathbf q_t\) |
| 在线未来力预测 | 必须 | 不需要 |
| per-axis residual scale | 需要 | 不需要 |
| conformal threshold | 需要 | 不需要 |
| 严格预测/测量延迟对齐 | 需要 | 只需正常时序同步 |
| π0 调用 | 触发后立即调用 | chunk 周期调用 |
| IK/FK | 不提供动作修正 | 完全不需要 |

代码已删除旧在线路径：

- force prediction residual；
- per-axis residual scale；
- conformal calibration threshold；
- trigger-and-discard broker；
- `return_force_prediction` 服务选项。

阶段一的 `F_phi` 仍可作为训练期辅助目标和离线诊断，但不再用于在线控制。

## 10. 实验与消融

主表建议比较：

1. Fixed Chunk；
2. Step-wise Replanning；
3. 旧 CoRACE baseline；
4. FRA。

报告：

- 任务成功率；
- 接触阶段成功率；
- 峰值力与力超限持续时间；
- 每个 episode 的 π0 调用次数；
- 平均 \(\|\delta\mathbf q_t\|\)；
- FRA 单步推理延迟。

关键消融：

- FRA without force：去掉 Contact Dynamics Token；
- FRA without stale-chunk training：只用当前 chunk 监督；
- FRA without physical anchor：使用未物理锚定的 force feature。

准确 claim：

> A lightweight two-timescale execution architecture that combines low-frequency VLA planning with high-frequency force-conditioned joint-space reflexes.

> FRA continuously corrects stale joint-action chunks using the latest contact feedback, providing closed-loop reactivity without force-error calibration, additional intervention data, or robot kinematics.

不要把 FRA 表述成经过认证的安全控制器，也不要声称它能处理任意分布外碰撞。力传感器硬限位、机器人原生安全控制与急停仍是独立安全层。

## 11. 参考依据

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)
- [TA-VLA](https://arxiv.org/abs/2509.07962)
- [Temporal Action Selection for Action Chunking](https://arxiv.org/abs/2511.04421)
- [Compliant Residual DAgger](https://arxiv.org/abs/2506.16685)
