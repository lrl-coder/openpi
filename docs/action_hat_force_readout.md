# Action-Hat Force Readout

本文记录当前 `F_phi` 的训练路径改动：训练时不再为了 force NLL 额外跑一次
`ActionExpert(obs, action_gt, t=0)`，而是复用 flow matching 的 denoising pass，
从模型当前预测的 clean action estimate 上做 force consequence readout。

## Motivation

旧版训练路径是：

```text
h_t = ActionExpert(obs, x_t, t)
v_t = action_out_proj(h_t)
L_FM = ||v_t - (noise - action_gt)||^2

h_0 = ActionExpert(obs, action_gt, 0)
F_phi(h_0) -> mu_f, log_sigma_f
L_force = NLL(force_targets; mu_f, sigma_f)
```

这样 `F_phi` 的监督语义很干净，但训练时 action expert 需要跑两遍。更重要的是，
如果后续希望在 diffusion / flow sampling 中使用 `F_phi`，采样中间步并没有
ground-truth clean action，只能拿到当前 noisy action `x_t` 和模型预测的 velocity `v_t`。

因此新版训练路径改成：

```text
h_t = ActionExpert(obs, x_t, t)
v_t = action_out_proj(h_t)
action_hat_0 = x_t - t * v_t

F_phi(h_t, stopgrad(action_hat_0)) -> mu_f, log_sigma_f
L_force = NLL(force_targets; mu_f, sigma_f)
```

这里的 `action_hat_0` 是当前 denoising step 隐含的 clean action estimate。它使
`F_phi` 的输入更接近推理期真实可获得的信息。

## Code Mapping

核心代码在 `src/openpi/models/pi0.py`：

```python
action_features = self._action_features_from_prefix_cache(observation, x_t, time, prefix_mask, kv_cache)
v_t = self.action_out_proj(action_features)
action_hat_0 = jax.lax.stop_gradient(x_t - time_expanded * v_t)
force_mu, force_log_sigma = self._decode_force_from_action_features(action_features, action_hat_0)
```

`_decode_force_from_action_features` 会把 `action_hat_0` 通过已有的 `action_in_proj`
投影到 action expert hidden width，然后与 denoising hidden feature 相加：

```python
action_hat_tokens = self.action_in_proj(action_hat_0.astype(action_features.dtype))
action_features = action_features + action_hat_tokens
```

因此当前 `F_phi` 的有效输入是：

```text
denoising action hidden feature h_t
+ projected predicted clean action action_hat_0
```

输出保持不变：

```text
mu_f:        [B, action_horizon, force_dim]
log_sigma_f:[B, action_horizon, force_dim]
```

默认 `force_dim = 6`。

## Gradient Choice

当前实现对 `action_hat_0` 使用 `stop_gradient`。原因是先保持训练稳定，让
`force_loss` 不直接通过 `action_hat_0 = x_t - t * v_t` 回传到 velocity head，
避免辅助 force objective 过早改写 flow matching 主目标。

但 `force_loss` 仍然会通过 `action_features` 更新 action expert hidden
representation，也会训练 `F_phi` 的 readout。后续如果需要更强的 force-action
coupling，可以做 ablation：

```text
stopgrad(action_hat_0)
vs
allow gradient through action_hat_0
vs
time-weighted force loss w(t) = (1 - t)^gamma
```

## Training And Inference Consistency

这个改动解决的是训练主路径中的不一致问题：

```text
训练: F_phi sees h_t and model-implied action_hat_0
推理采样中: h_t, v_t, action_hat_0 are also available at every denoising step
```

当前代码的在线 policy 仍然在采样结束后调用 `predict_force(observation, actions)`，
用于缓存下一轮 force residual。这个离线预测路径会用最终动作 `actions` 作为
`action_hat_0`，语义上等价于 `t=0` 的 clean action readout。

后续如果要把 `F_phi` 重新接入 sampling-loop guidance，可以直接在每个采样步复用：

```text
h_t, v_t -> action_hat_0 -> F_phi(h_t, action_hat_0)
```

不需要再额外跑 clean action expert pass。

