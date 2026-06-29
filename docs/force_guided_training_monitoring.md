# Force-guided pi0 训练监视文档

本文档用于监视 `pi0_flexiv_pump_1bottle_inputForce_lora_force_guided` / TCN force semantic encoder 训练是否异常。

## 1. 启动阶段检查

训练命令示例：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=TCN_flexiv_pump_lora_force_guided \
  --overwrite
```

正常现象：

- 日志中可能出现 XLA rematerialization warning，例如显存优化无法进一步降低。这不是训练崩溃原因。
- checkpoint 目录下会生成：
  - `metrics.csv`
  - `wandb_id.txt`
  - `force_prediction_traces/`

异常现象：

- 如果再次出现：

```text
pjit out_shardings.model_def
different pytree metadata
```

说明 `init_train_state` 的 `eval_shape` 结构和实际加载 checkpoint 后的结构不一致，需要检查 `scripts/train.py` 里是否已经用 `partial_params` 重新计算 `train_state_shape` 和 `state_sharding`。

## 2. 总 loss 与 force NLL

重点看 `metrics.csv` 或 W&B 中这些字段：

```text
loss
loss_fm
loss_force_nll
loss_force_weighted
force_nll
force_nll_negative_frac
```

判断规则：

- `loss_force_nll` 变成负值是正常的。连续高斯 NLL 可以小于 0。
- 更重要的是看它是否和残差、方差一起健康变化。
- 如果 `loss_force_nll` 很负，但预测残差没有下降，可能是方差塌缩造成的虚假好看。

需要重点联动观察：

```text
force_residual_mse_mean
force_pred_log_sigma_mean
force_pred_log_sigma_min_clip_frac
force_pred_log_sigma_max_clip_frac
force_pred_sigma_to_true_std_ratio_mean
```

异常信号：

- `force_pred_log_sigma_min_clip_frac` 长期接近 1：模型把方差压到下限，可能过度自信。
- `force_pred_sigma_to_true_std_ratio_mean` 长期远小于 1：预测方差明显低估真实波动。
- `force_residual_mse_mean` 不降，但 `loss_force_nll` 继续变好：需要警惕 sigma 学坏。

建议处理：

- 提高 `log_sigma` 下限，例如从 `-5.0` 调到 `-4.0` 或 `-3.0`。
- 给 `log_sigma` 加正则，避免无限压小方差。
- 降低 `force_loss_weight`，防止 force head 主导训练。

## 3. 未来 50 步 force 预测轨迹

训练每个 `log_interval` 会在以下目录输出：

```text
<checkpoint_dir>/force_prediction_traces/
```

典型文件：

```text
step_00000000.csv
step_00000000.npz
step_00000200.csv
step_00000200.npz
step_00000200_semantic_features.csv
```

`step_xxxxxxxx.csv` 的列：

```text
sample,time,axis,target,mu,log_sigma,sigma,var
```

含义：

- `target`：真实未来 force
- `mu`：模型预测均值
- `sigma`：模型预测标准差
- `var`：模型预测方差
- `time`：未来 action horizon 内的时间步，通常 0 到 49
- `axis`：6 维力/力矩轴

建议画图方式：

- 对每个 `sample` 和 `axis`，画 `target` 与 `mu` 随 `time` 的曲线。
- 同时画 `mu ± sigma` 或 `mu ± 2*sigma`，看真实值是否落在合理不确定性范围内。

异常信号：

- `mu` 对所有时间步几乎是一条水平线，但 `target` 明显变化：模型没有学到未来 force 动态。
- `sigma` 全部极小，而 `mu` 偏差明显：过度自信。
- `sigma` 全部极大：模型用大方差逃避预测。
- step 增加后，`mu` 仍然完全不跟随 `target`：force head 训练信号可能太弱，或 action hidden feature 与 force target 对不上。

快速分析代码：

```python
import pandas as pd
import matplotlib.pyplot as plt

path = "/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/TCN_flexiv_pump_lora_force_guided/force_prediction_traces/step_00000200.csv"
df = pd.read_csv(path)

sample = 0
axis = 0
d = df[(df["sample"] == sample) & (df["axis"] == axis)]

plt.plot(d["time"], d["target"], label="target")
plt.plot(d["time"], d["mu"], label="mu")
plt.fill_between(d["time"], d["mu"] - d["sigma"], d["mu"] + d["sigma"], alpha=0.2, label="mu±sigma")
plt.legend()
plt.grid(True)
plt.show()
```

## 4. Semantic query 与 TCN force feature 是否 collapse

我们重点担心：

```text
query_feature = VLM semantic query head 输出
force_feature = TCN(force_history_global) 输出
```

如果二者都输出常量，`1 - cosine` 也可能接近 0，但没有真实语义信息。

每个 log step 会输出：

```text
step_xxxxxxxx_semantic_features.csv
```

列：

```text
sample,dim,query_feature,force_feature
```

W&B 中还会记录：

```text
force_semantic_features/query_batch_std_mean
force_semantic_features/force_batch_std_mean
force_semantic_features/cosine_mean
force_semantic_features/cosine_std
```

判断规则：

- `query_batch_std_mean` 接近 0：不同输入下 query feature 几乎不变，可能 collapse。
- `force_batch_std_mean` 接近 0：不同 force history 下 TCN feature 几乎不变，可能 collapse。
- 两个 std 都接近 0，同时 `cosine_mean` 很高：高度怀疑两个头学成同一个常量方向。
- `cosine_std` 接近 0：所有样本的对齐程度几乎一样，也要警惕。

建议阈值：

- `< 1e-4`：强烈怀疑常量输出。
- `1e-4 ~ 1e-3`：需要结合曲线和 batch 多样性判断。
- `> 1e-3`：通常说明至少不是完全常量，但仍要看是否有语义区分度。

快速分析代码：

```python
import pandas as pd

path = "/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/TCN_flexiv_pump_lora_force_guided/force_prediction_traces/step_00000200_semantic_features.csv"
df = pd.read_csv(path)

query = df.pivot(index="sample", columns="dim", values="query_feature").to_numpy()
force = df.pivot(index="sample", columns="dim", values="force_feature").to_numpy()

print("query batch std mean:", query.std(axis=0).mean())
print("force batch std mean:", force.std(axis=0).mean())
print("query sample norm:", (query**2).sum(axis=1) ** 0.5)
print("force sample norm:", (force**2).sum(axis=1) ** 0.5)
```

## 5. TCN encoder 训练状态

当前 TCN force semantic encoder：

- 不是 stop-gradient。
- 权重初始来自随机正态初始化，`stddev=0.01`。
- base checkpoint 中没有这些新增 force 参数，loader 会用当前随机初始化补齐 `force_*` 参数。
- LoRA freeze 不会冻住 `force_semantic_tcn`，所以它是可训练的。

它主要由以下 loss 训练：

```text
loss_force_semantic_align = 1 - cosine(query_feature, force_feature)
```

异常信号：

- `loss_force_semantic_align` 快速降到很低，但两个 batch std 都接近 0：可能是常量 collapse。
- `loss_force_semantic_align` 长期不降，且 force feature 有变化：query 侧可能没有学会对齐。
- `loss_force_semantic_align` 长期不降，且 force feature std 接近 0：TCN 可能没有获得有效训练信号。

## 6. 推荐的日常监控顺序

每次训练建议按下面顺序看：

1. 启动是否正常：没有 `pjit out_shardings.model_def` 崩溃。
2. `loss_fm` 是否正常下降：主动作模型不能被 force loss 破坏。
3. `loss_force_nll` 和 `force_residual_mse_mean` 是否同步改善。
4. `force_pred_log_sigma_min_clip_frac` 是否长期贴近 1。
5. 打开 `force_prediction_traces/step_xxxxxxxx.csv`，看未来 50 步 `target` vs `mu`。
6. 看 `query_batch_std_mean` 和 `force_batch_std_mean`，判断 semantic 两个头是否 collapse。
7. 如果发现 semantic collapse，不要只看 cosine；优先加物理预测辅助任务，例如 force summary prediction。

## 7. 常见异常与处理表

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `loss_force_nll < 0` | 连续高斯 NLL 正常现象 | 不单独视为异常，继续看 sigma 和 residual |
| `log_sigma_min_clip_frac -> 1` | 方差塌缩，过度自信 | 提高 log_sigma 下限或加正则 |
| `mu` 是水平线 | force head 没学到动态 | 检查 force_loss_weight、action/force 时间对齐 |
| `sigma` 很大 | 用不确定性逃避预测 | 降低 log_sigma 上限或加强均值监督 |
| semantic cosine 很高但 std 很低 | 两个头常量 collapse | 加 force summary 辅助任务，或只对 query 侧做 alignment |
| `loss_fm` 变差明显 | force 辅助任务干扰主任务 | 降低 force_loss_weight / force_target_loss_weight |

## 8. 当前最重要的观察点

现阶段最值得看的不是单一 loss，而是这三组是否一致：

```text
force trajectory:
target vs mu vs sigma

force uncertainty:
residual_mse vs pred_sigma vs true_std

semantic features:
query_batch_std_mean vs force_batch_std_mean vs cosine_mean
```

如果这三组都健康，训练基本可信；如果 loss 很好但轨迹或特征异常，优先相信轨迹和特征诊断。
