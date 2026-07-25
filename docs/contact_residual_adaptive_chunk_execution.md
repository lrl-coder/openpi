# CoRACE：Contact-Residual Adaptive Chunk Execution

## 定位

CoRACE 是 force-aware π0 的轻量推理执行模块。模型仍按原方式生成 action chunk；辅助头
`F_phi` 同时预测该 chunk 的未来 6D 力/力矩。机器人每执行一步，就用新测得的力验证该步预测。
如果误差超过离线标定阈值，执行端立即丢弃尚未执行的 action suffix，并基于最新观测重新推理。

它只改变“一个 chunk 执行多久”，不修改 token、attention、flow-matching 采样或模型参数：

```text
predict chunk -> execute one step -> verify force consequence
                                  -> pass: continue chunk
                                  -> fail: discard suffix and replan
```

这使论文中的因果链条保持单一：

```text
预测接触后果与真实接触后果不一致
    -> 当前 action chunk 已经变 stale
    -> 缩短本次实际执行 horizon
    -> 从最新观测重新规划
```

## 1. 严格时间对齐

时刻 `t` 的策略输出：

```text
A_t = [a_t,0, ..., a_t,H-1]
F_hat_t = [f_hat_t,0, ..., f_hat_t,H-1]
```

执行 `a_t,k` 并取得其对应传感器读数 `f_t,k` 后，只计算同索引残差：

```text
f_t,k <-> f_hat_t,k
```

当前 Flexiv 数据管线的 action timestamps 是 `t ... t+H-1`，`force_targets` 来自
`t+1 ... t+H` 的 state，因此 `f_hat_t,k` 的语义是 **执行 `a_t,k` 后的力**。在线 broker 也只在动作
已经下发、下一帧观测到达后比较索引 `k`，训练与执行约定一致。若机器人日志把力记录在命令前而不是命令后，
必须先改数据对齐，不能只改 broker 索引。

若控制循环已经发出了 `n` 个动作，已知传感器相对命令存在 `d` 个控制步延迟，则当前测量对应：

```text
k = n - 1 - d
```

实现中的 `measurement_delay_steps=d` 显式编码这件事。`d` 必须通过时间戳、阶跃响应或互相关实验确定，
不能用阈值掩盖时序误差。

## 2. 无量纲残差

不同力/力矩轴的单位和动态范围不同，不能直接把 6 轴 raw MSE 相加。CoRACE 定义：

```text
r_t,k = (1 / D) * sum_j (
    (f_t,k,j - f_hat_t,k,j) / s_j
) ** 2
```

其中 `D=6`，`s_j` 是每个轴的残差尺度。推荐用独立的训练划分估计 RMS：

```text
s_j = max(
    epsilon,
    sqrt(mean_train((f_j - f_hat_j) ** 2))
)
```

代码支持全 horizon 共用的 `[D]` 尺度，也支持随预测步变化的 `[H, D]` 尺度。后者适合预测误差随
horizon 明显增大的模型。

## 3. Split-conformal 阈值

阈值不使用手调常数。对不参与模型训练和尺度估计的 nominal calibration chunks，先计算每一步残差，
再把每个 chunk 的最大值作为 calibration score：

```text
S_i = max_k r_i,k
```

给定目标误触发率 `alpha`，令：

```text
rank = ceil((N + 1) * (1 - alpha))
q = rank-th smallest value of {S_i}
```

需要 `rank <= N`；等价地，至少准备 `ceil(1 / alpha) - 1` 个 calibration chunks，否则实现会拒绝给出
一个伪装成有限样本保证的阈值。

在线触发条件为：

```text
r_t,k > q
```

使用 chunk 最大值而不是单步分位数，是为了直接标定“一个正常 chunk 内至少发生一次错误中断”的概率。
在 calibration chunk 与部署 chunk 可交换的条件下，这是标准 split-conformal 的有限样本阈值。
机器人连续轨迹通常不完全可交换，因此论文中应把它表述为 **nominal false-interruption calibration**，
不要声称它对任意分布漂移提供安全保证。建议按 episode 划分 calibration 数据，避免相邻窗口同时落入
训练集和 calibration 集。

## 4. 在线算法

```text
input: policy, threshold q, force scale s, maximum execution horizon H_exec

request (A, F_hat) from current observation
for k in 0 ... H_exec - 1:
    issue A[k]
    receive the next observation and measured force
    align the measurement to F_hat[k] (with calibrated sensor delay)
    compute normalized residual r_k
    if r_k > q:
        discard A[k+1:]
        request a new (A, F_hat) from the latest observation
```

首个动作前没有已执行后果，因此不计算残差。CoRACE 必须在每个低层控制步调用；网络请求只会在 chunk
耗尽或触发时发生。

## 5. 代码接口

服务端启用力预测输出：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --policy.dir=/path/to/checkpoint \
  --policy.return-force-prediction
```

服务端返回：

```text
actions           [H, action_dim]
force_prediction  [H, 6]  # 已反归一化到传感器物理单位
```

客户端包装 websocket policy：

```python
import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
policy = action_chunk_broker.CoRACEActionChunkBroker(
    client,
    action_horizon=8,
    residual_threshold=calibrated_threshold,
    force_scale=force_scale,
    force_key="force",
    measurement_delay_steps=0,
)

for _ in range(num_control_steps):
    observation = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state_with_force,
        "force": np.asarray(measured_force),
        "prompt": instruction,
    }
    result = policy.infer(observation)
    robot.execute(result["actions"])
    log(result["corace"])
```

离线标定：

```python
from openpi_client import action_chunk_broker

# scale split: [N_train, H, 6]
force_scale = action_chunk_broker.estimate_force_scale(
    scale_observed_force,
    scale_predicted_force,
)

# disjoint calibration split: [N_cal, H, 6]
threshold = action_chunk_broker.calibrate_corace_threshold(
    calibration_observed_force,
    calibration_predicted_force,
    force_scale,
    alpha=0.05,
)
```

`result["corace"]` 包含：

```text
triggered
residual_score
residual_threshold
prediction_index
replan_count
```

## 6. 实验与消融

主表至少比较：

1. fixed chunk：固定执行 `H_exec`；
2. step-wise replanning：每步都重新推理，作为最高计算量闭环基线；
3. CoRACE：残差触发的自适应执行。

核心指标：

- 任务成功率与接触阶段成功率；
- 峰值力/力矩、超安全阈值时长；
- 每个 episode 的 policy query 次数与平均实际 chunk 长度；
- false interruption rate、trigger precision、触发到恢复的步数；
- `F_phi` 在 horizon 各步、各轴的 RMSE。

必要消融：

- raw MSE vs 轴尺度归一化残差；
- 手调阈值 vs held-out conformal 阈值；
- 错误时间索引/不补偿传感器延迟 vs 正确对齐；
- 随机 trigger（匹配相同 query budget），排除“只是多推理几次”的解释。

## 7. 参考依据与方法边界

- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 将并行解码、连续动作和 action chunking
  组合成简洁有效的微调方案，支持保留 chunk 的推理效率收益。
- [ACT](https://arxiv.org/abs/2304.13705) 是机器人模仿学习中 action chunking 的直接基础。
- [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161) 说明固定执行 horizon 的
  reactivity/consistency 权衡可以在推理期动态处理。
- [VLA-Corrector](https://arxiv.org/abs/2607.01804) 用预测与真实视觉演化偏差触发 chunk 截断；
  CoRACE 采用同类“future-reality verification”执行范式，但监测量是与接触直接相关的 6D 力后果。
- [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)
  给出 split-conformal 有限样本分位数构造。
- [Sequential Predictive Conformal Inference](https://proceedings.mlr.press/v202/xu23r.html)
  说明时间序列非可交换性会削弱普通 conformal 假设，因此连续部署应同时报告经验误触发率。

CoRACE 不是底层阻抗/导纳控制器，也不是经过认证的安全屏障。力传感器硬限位、急停和机器人原生安全控制
仍应独立存在。它的论文 claim 应限定为：**利用动作条件力预测误差，对 VLA action chunk 进行事件触发的
自适应截断与重规划。**
