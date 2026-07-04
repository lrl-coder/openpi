# Force-Residual Attention Modulation (FRAM) 推理使用说明

## 1. 一句话说明

FRAM 只在推理闭环中启用。它会缓存上一轮动作对应的未来力预测序列；下一轮收到新的真实力传感器数据后，比较“已经执行时间步”的预测力和真实力。如果残差大，就在本轮 Action expert 前向中增强 local force token，使动作生成更关注力反馈。

FRAM 不构造未来力目标，也不需要知道未来真实力。

## 2. 使用前提

必须满足以下条件：

1. 使用 force-guided 模型配置，例如：

   - `pi0_flexiv_pump_1bottle_inputForce_lora_force_guided`
   - `pi0_flexiv_pump_1bottle_inputForce_lora_with_force_guided`
   - `pi0_flexiv_pump_1bottle_inputForce_force_guided`
   - `pi0_flexiv_pump_1bottle_inputForce_with_force_guided`

2. 模型配置中 `force_guidance=True`，这样模型才会启用：

   - local force token
   - $F_\phi$ force consequence predictor
   - FRAM residual-to-attention modulation

3. 推理输入中必须能得到当前力 `force`。

   对 Flexiv pump 数据流来说，输入原始 13 维 `observation.state` 即可；transform 会从 `state[7:13]` 提取当前 force/torque。

4. JAX 模型路径才支持当前 FRAM 在线闭环。PyTorch policy 路径目前不会调用 `predict_force`，因此不会缓存上一轮力预测。

## 3. 最常用启动方式

启动 policy server 时加：

```bash
uv run scripts/serve_policy.py \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --policy.dir=/path/to/checkpoint/step \
  --policy.force-attention-modulation-from-residual
```

旧参数仍然兼容：

```bash
--policy.force-guidance-from-residual
```

但论文和新实验记录中建议使用新名字：

```bash
--policy.force-attention-modulation-from-residual
```

## 4. 推理时序

FRAM 的在线时序如下：

1. 第一次推理：

   - 当前没有上一轮力预测。
   - FRAM 不会调制 force token。
   - 模型正常生成动作，并调用 `predict_force` 缓存本轮动作的未来力预测：

     $$
     \left\{
     \mu_{t+h|t},
     \Sigma_{t+h|t}
     \right\}_{h=1}^{H}
     $$

2. 第二次及之后推理：

   - policy 收到新的真实力传感器读数。
   - 用真实力和上一轮预测力前缀计算标准化残差：

     $$
     r_t
     =
     \frac{1}{m}
     \sum_{h=1}^{m}
     \left(
     F^{obs}_{t-h+1}
     -
     \mu_{h}
     \right)^\top
     \Sigma_h^{-1}
     \left(
     F^{obs}_{t-h+1}
     -
     \mu_h
     \right)
     $$

   - 将残差映射成 attention modulation：

     $$
     g_t
     =
     g_{\max}
     \frac{r_t}{r_t+d_f}
     $$

   - 在本轮 Action expert 中调制 local force token：

     $$
     \tilde{z}^{force}_t
     =
     \left(
     1+g_t
     \right)
     z^{force}_t
     $$

3. 如果传入 `reset=True`：

   - policy 会清空上一轮力预测缓存。
   - 下一次推理重新从无 FRAM 调制开始。

## 5. 关键配置项

### 5.1 `force_attention_modulation_from_residual`

是否启用 FRAM 在线闭环。

命令行：

```bash
--policy.force-attention-modulation-from-residual
```

Python API：

```python
sample_kwargs={
    "force_attention_modulation_from_residual": True,
}
```

### 5.2 `force_attention_modulation_max`

最大 force attention modulation 强度，也就是公式中的 $g_{\max}$。

推荐初始值：

```python
force_attention_modulation_max = 0.2
```

当前代码保留旧字段兼容：

```python
force_guidance_lambda_max = 0.2
```

如果 `force_attention_modulation_max` 显式设置，则优先使用它；否则回退到 `force_guidance_lambda_max`。

### 5.3 `force_residual_window`

用于计算残差的已执行力窗口长度 $m$。

默认：

```python
force_residual_window = 1
```

也就是只比较上一轮第一个预测力和当前真实力。这是最稳妥的在线设置。

如果你的部署侧能可靠提供最近多个已执行时间步的 `force_history_local`，可以增大窗口，例如：

```python
sample_kwargs={
    "force_attention_modulation_from_residual": True,
    "force_residual_window": 4,
}
```

注意：窗口越大，对时序对齐要求越高。如果控制频率、动作执行步数和传感器采样不同步，建议保持 `1`。

## 6. Python API 示例

如果不通过 `serve_policy.py`，而是直接创建 policy：

```python
from openpi.training import config as _config
from openpi.policies import policy_config

cfg = _config.get_config("pi0_flexiv_pump_1bottle_inputForce_lora_force_guided")

policy = policy_config.create_trained_policy(
    cfg,
    "/path/to/checkpoint/step",
    sample_kwargs={
        "force_attention_modulation_from_residual": True,
        "force_residual_window": 1,
    },
)
```

然后每次推理输入至少需要包含原始观测。Flexiv transform 会从 13 维 state 中提取力：

```python
obs = {
    "state": state_13d,
    "images": images,
    "prompt": prompt,
}

action = policy.infer(obs)["actions"]
```

如果你已经在外部维护了力历史，也可以显式传入：

```python
obs = {
    "state": state_13d,
    "force": current_force_6d,
    "force_history_local": force_history_local,   # shape: [T, 6]
    "images": images,
    "prompt": prompt,
}
```

## 7. 推荐默认配置

建议先使用：

```python
force_attention_modulation_from_residual = True
force_residual_window = 1
force_attention_modulation_max = 0.2
```

如果残差触发后动作波动较大，降低：

```python
force_attention_modulation_max = 0.05 ~ 0.1
```

如果力反馈被模型忽略，且动作仍明显不响应接触变化，提高：

```python
force_attention_modulation_max = 0.3 ~ 0.5
```

不建议一开始使用过大的值，因为 FRAM 是对 force token 的条件强度调制，过大可能让策略过度依赖瞬时力噪声。

## 8. 调试检查

### 8.1 第一轮没有 FRAM 是正常的

第一轮没有上一轮力预测，因此 `force_prediction_error` 不会被传入 `sample_actions`。从第二轮开始才会生效。

### 8.2 确认输入中有力

如果 transform 后没有 `force`，FRAM 无法计算残差。

Flexiv 任务中应保证输入 `observation.state` 是 13 维，且后 6 维是 force/torque。

### 8.3 确认模型是 force-guided 配置

如果配置不是 `*_force_guided`，则：

```python
force_guidance = False
```

local force token 和 $F_\phi$ 都不会启用，FRAM 也不会生效。

### 8.4 reset 会清空缓存

如果每步都传入：

```python
reset=True
```

FRAM 会一直处在“第一轮无上一轮预测”的状态。只应在 episode 开始或机器人状态需要重置时传 `reset=True`。

## 9. 推荐论文/实验表述

可以写：

> During inference, FRAM is enabled after the first control cycle. The policy caches the force consequence predicted for the previously issued action chunk. Once new force measurements are observed, FRAM computes a normalized force-residual score over the executed steps and converts it into a bounded attention modulation coefficient for the local force token in the next Action Expert forward pass.

中文：

> 推理时，FRAM 从第二个控制周期开始生效。策略缓存上一轮动作 chunk 的力后果预测；当新的真实力测量到达后，FRAM 在已执行时间步上计算标准化力残差，并将其转换为有界的 attention modulation 系数，用于调制下一轮 Action expert 中的 local force token。
