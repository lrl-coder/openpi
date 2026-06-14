# flexiv_pump_1bottle_inputForce 微调 pi0_base

本文档记录当前机器上使用 openpi 微调 `pi0_base` 的运行方式。数据集已经是 LeRobot dataset 格式：

```text
/root/autodl-tmp/data/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce
```

当前已在 `src/openpi/training/config.py` 中加入 baseline 和 force-guided 训练配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora
pi0_flexiv_pump_1bottle_inputForce_lora_with_force
pi0_flexiv_pump_1bottle_inputForce_lora_force_guided
pi0_flexiv_pump_1bottle_inputForce_lora_with_force_guided
pi0_flexiv_pump_1bottle_inputForce
pi0_flexiv_pump_1bottle_inputForce_with_force
pi0_flexiv_pump_1bottle_inputForce_force_guided
pi0_flexiv_pump_1bottle_inputForce_with_force_guided
```

推荐优先使用 `pi0_flexiv_pump_1bottle_inputForce_lora`。openpi README 中给出的显存参考是：LoRA 微调约需要 22.5GB 以上显存，全量微调约需要 70GB 以上显存。
如果要训练新增 CST/力预测/语义力目标方案，优先使用 `pi0_flexiv_pump_1bottle_inputForce_lora_force_guided`。

## 训练配置在哪里看

主要看这个文件：

```text
/root/autodl-tmp/openpi/src/openpi/training/config.py
```

当前 flexiv pump 相关配置在 `_CONFIGS` 列表里：

```text
pi0_flexiv_pump_1bottle_inputForce_lora            # LoRA, 默认只用 state 前 7 维
pi0_flexiv_pump_1bottle_inputForce_lora_with_force # LoRA, 使用完整 13 维 state
pi0_flexiv_pump_1bottle_inputForce_lora_force_guided            # LoRA, 前 7 维 state + 单独 force 辅助头
pi0_flexiv_pump_1bottle_inputForce_lora_with_force_guided # LoRA, 完整 13 维 state + 单独 force 辅助头
pi0_flexiv_pump_1bottle_inputForce                 # 全量微调, 默认只用 state 前 7 维
pi0_flexiv_pump_1bottle_inputForce_with_force      # 全量微调, 使用完整 13 维 state
pi0_flexiv_pump_1bottle_inputForce_force_guided                 # 全量微调, 前 7 维 state + 单独 force 辅助头
pi0_flexiv_pump_1bottle_inputForce_with_force_guided      # 全量微调, 完整 13 维 state + 单独 force 辅助头
```

这些配置当前显式设置了：

```text
num_train_steps = 20_000
batch_size      = 16
num_workers     = 0
weight_loader   = gs://openpi-assets/checkpoints/pi0_base/params
```

force-guided 配置额外设置：

```text
model.force_guidance             = True
model.force_loss_weight          = 0.05
model.force_target_loss_weight   = 0.01
model.force_guidance_lambda_max  = 0.2
model.force_guidance_k           = 1.0
model.force_guidance_tau0        = 6.0
```

LoRA 配置额外设置：

```text
model         = Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
freeze_filter = LoRA 默认 freeze filter
ema_decay     = None
```

全量微调配置使用：

```text
model     = Pi0Config()
ema_decay = 0.99  # 继承 TrainConfig 默认值
```

没有在 flexiv pump 配置里显式写的参数，会继承 `TrainConfig` 默认值。常用默认值如下：

```text
project_name        = openpi
assets_base_dir     = ./assets
checkpoint_base_dir = /root/autodl-fs/openpi_checkpoints
seed                = 42
log_interval        = 100
save_interval       = 1000
keep_period         = 5000
wandb_enabled       = True
fsdp_devices        = 1
```

也就是说，默认每 100 step 打印/记录一次日志，每 1000 step 保存一次 checkpoint，且每 5000 step 的 checkpoint 会被保留。终端只打印核心摘要，完整 scalar 指标会写入 checkpoint 目录下的 `metrics.csv`，同时继续写入 wandb。

如果想临时覆盖配置，不一定要改代码，可以在训练命令后加参数。例如只训练 5000 step、关闭 wandb：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora \
  --exp-name=flexiv_pump_lora_5k \
  --num-train-steps=5000 \
  --no-wandb-enabled \
  --overwrite
```

也可以用 `--help` 查看某个 config 当前所有可覆盖参数：

```bash
uv run python scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora --help
```

## 数据映射

配置里使用的字段映射如下：

```text
observation.image       -> base_0_rgb
observation.wrist_image -> left_wrist_0_rgb
observation.state       -> state
action                  -> actions
tasks.jsonl             -> prompt
```

`observation.state` 原始是 13 维：

```text
current_eef_pose(6), gripper_width(1), f_ext_base_frame(6)
```

默认训练只使用前 7 维作为 state：

```text
current_eef_pose(6), gripper_width(1)
```

最后 6 维 `f_ext_base_frame` 先不参与训练。需要加入力信息时，使用 `*_with_force` 配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora_with_force
pi0_flexiv_pump_1bottle_inputForce_with_force
```

force-guided 配置会始终从原始 13 维 `observation.state` 里拆出最后 6 维作为单独的 force 信号：

```text
force              = observation.state[7:13]
force_history      = 因果历史 force 窗口, shape = (16, 6)
force_targets      = 下一时刻 force 序列, shape = (50, 6)
force_task_target  = 当前 action chunk 内 force_targets 的均值, shape = (6,), 仅作为兼容字段
```

`*_force_guided` 默认仍只把前 7 维送进 pi0 的常规 `state` token；力通过 `force_history` 的双路径编码进入 VLM prefix 和动作头 suffix，并通过 `F_phi`、semantic target head、CST 引导影响训练和推理。`*_with_force_guided` 则同时把完整 13 维送进常规 `state` token，用于做对照实验。

注意：所有 `*_force_guided` 配置在训练和推理时都要求输入原始 13 维 `observation/state`，因为当前力 `force = state[7:13]` 会被用于构造 `force_history`、CST 和 `F_phi`。区别只在于常规 pi0 state token 使用前 7 维还是完整 13 维。

pi0 需要 32 维 state/action，所以 openpi 会把 7 维或 13 维 state 自动 pad 到 32 维。`action` 原始是 7 维：

```text
target_eef_pose(6), target_gripper_width(1)
```

训练时前 6 维 EEF pose 会转为 delta action，最后 1 维 gripper width 保持绝对值。推理输出会自动转回 7 维动作。

## 环境变量

每次训练前先进入 openpi 目录并设置环境变量：

```bash
cd /root/autodl-tmp/openpi

export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-fs/openpi_cache/hf_datasets
export HF_HOME=/root/autodl-fs/openpi_cache/hf_home
```

不建议把 `HF_DATASETS_CACHE` 放在 `/root/autodl-tmp`。LeRobot 会把 parquet 生成 Arrow cache，当前数据集生成过程已经在小盘上触发过 `No space left on device`。如果确认空间足够，才临时改回：

```bash
cd /root/autodl-tmp/openpi

export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-tmp/openpi/.cache/huggingface/datasets
export HF_HOME=/root/autodl-tmp/openpi/.cache/huggingface
```

注意：当前这个本地 openpi 仓库里，预训练权重下载缓存实际默认解析到：

```text
/root/autodl-fs/openpi
```

也就是 `pi0_base` 会从 `gs://openpi-assets/checkpoints/pi0_base/params` 下载/缓存到这个大盘目录；不需要额外放到项目 `.cache/openpi` 里。

## 归一化统计

训练前必须先有 norm stats。目前四个配置的 norm stats 都已经准备好，可以直接训练：

```text
/root/autodl-tmp/openpi/assets/pi0_flexiv_pump_1bottle_inputForce_lora/flexiv_pump_1bottle_inputForce/norm_stats.json
/root/autodl-tmp/openpi/assets/pi0_flexiv_pump_1bottle_inputForce_lora_with_force/flexiv_pump_1bottle_inputForce/norm_stats.json
/root/autodl-tmp/openpi/assets/pi0_flexiv_pump_1bottle_inputForce/flexiv_pump_1bottle_inputForce/norm_stats.json
/root/autodl-tmp/openpi/assets/pi0_flexiv_pump_1bottle_inputForce_with_force/flexiv_pump_1bottle_inputForce/norm_stats.json
```

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

## LoRA 微调

推荐命令：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora \
  --exp-name=flexiv_pump_lora \
  --overwrite
```

如果训练中断后想继续同一个实验：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora \
  --exp-name=flexiv_pump_lora \
  --resume
```

checkpoint 默认保存到：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora/flexiv_pump_lora
```

这个目录由两部分决定：

```text
checkpoint_base_dir = /root/autodl-fs/openpi_checkpoints
config name         = pi0_flexiv_pump_1bottle_inputForce_lora
--exp-name          = flexiv_pump_lora
```

公式是：

```text
{checkpoint_base_dir}/{config_name}/{exp_name}
```

在当前机器上，`/root/autodl-fs` 是指向 AutoDL 文件存储的路径；Python 解析后的绝对路径可能显示为 `/autodl-fs/data/...`，两者指向同一个文件存储。

如果命令里把 `--exp-name` 改成 `test_run`，保存目录就会变成：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora/test_run
```

## 全量微调

只有显存足够时再用全量微调：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce \
  --exp-name=flexiv_pump_full \
  --overwrite
```

checkpoint 默认保存到：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce/flexiv_pump_full
```

## 带力信息训练

如果要把 `f_ext_base_frame(6)` 也加入 state，使用 `*_with_force` 配置。LoRA 命令：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_flexiv_pump_1bottle_inputForce_lora_with_force

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora_with_force \
  --exp-name=flexiv_pump_lora_with_force \
  --overwrite
```

推理时传入的 `observation/state` 也必须是 13 维：

```text
current_eef_pose(6), gripper_width(1), f_ext_base_frame(6)
```

带力 LoRA 的 checkpoint 默认保存到：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_with_force/flexiv_pump_lora_with_force
```

## Force-guided 训练

新增方案对应四个配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora_force_guided
pi0_flexiv_pump_1bottle_inputForce_lora_with_force_guided
pi0_flexiv_pump_1bottle_inputForce_force_guided
pi0_flexiv_pump_1bottle_inputForce_with_force_guided
```

推荐先跑 LoRA 版本：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --exp-name=flexiv_pump_lora_force_guided \
  --overwrite
```

该配置复用已有 baseline norm stats，并自动从 `state[7:13]` 派生 `force`、`force_history`、`force_targets`、`force_task_target` 的归一化统计，不需要单独复制一份 norm stats。

当前实现包含：

```text
force_history: 16 x 6, 对应当前帧及过去 15 帧
VLM slow path: force_history -> 4 个 force patch tokens -> VLM prefix
Action fast path: force_history -> causal dilated Conv1D -> force condition token -> action suffix
VLM semantic force target head: C_pool(image, language, force tokens) -> (mu*, log_sigma*)
Action auxiliary force predictor F_phi: action-head hidden features -> (mu_f, log_sigma_f)
Training loss: L = L_FM + 0.05 * L_F_phi + 0.01 * L_target
Policy CST: tau = sum((f_t - mu_{f,t-1})^2 / sigma_{f,t-1}^2)
Guided sampling: lambda(tau) = 0.2 * sigmoid(1.0 * (tau - 6.0))
```

训练监督里，`F_phi` 不再是独立拼接 MLP；它挂在原动作头的 action-token hidden features 后面，只做数值解码。标签仍使用 action chunk 中每个示范动作对应的下一帧 force。
VLM semantic force target head 的 `L_target` 当前优先使用整段 `force_targets` 作为 NLL 目标，而不是单点 `force_task_target`。也就是说，`(mu*, sigma*)` 对 `(B, 50, 6)` 的力序列广播，监督目标是 chunk 内力分布：

```text
L_target = mean_t NLL(force_targets[:, t, :] ; mu*, sigma*)
```

这样 `sigma*` 的最优解对应 force 序列的经验方差，而不是 `mu*` 对单点均值的预测残差方差。只有当 `force_targets` 不存在时，代码才会回退到 legacy 的 `force_task_target` 单点监督。

### Force NLL / 方差诊断

force-guided 训练会额外在日志 step 跑一次无梯度诊断前向，并把力预测分布指标写到 checkpoint 目录下的 `metrics.csv` 和 wandb。终端 tqdm 只显示核心摘要，训练反向仍只走原来的 `train_step`，诊断不进入梯度图。

连续高斯 NLL 可以是负数，这本身不是数值错误。因为 NLL 里有 `log(sigma)` 项，当力标签已经归一化、预测残差很小、模型预测的 `sigma < 1` 时，`L_F_phi` 或 `L_target` 可以小于 0。需要判断的是方差是否坍塌。旧版本曾用 `force_task_target` 单点均值监督 `L_target`，这会让 `sigma*` 学到均值预测残差而不是示范力分布方差；现在已经改为优先用整段 `force_targets` 监督。

重点看这些指标：

```text
loss                 # 总训练 loss
diagnostic_loss      # 同一 batch 的诊断总 loss
loss_fm              # flow matching loss
loss_force_nll       # F_phi 原始 NLL，未乘 0.05
loss_force_weighted  # 0.05 * loss_force_nll
loss_force_target_nll
loss_force_target_weighted

force_pred_log_sigma_mean/min/max
force_pred_log_sigma_min_clip_frac
force_pred_sigma_mean/min/max
force_pred_var_mean
force_true_var_mean
force_pred_var_minus_true_var_mean
force_pred_var_abs_gap_mean
force_residual_mse_mean
force_pred_sigma_to_true_std_ratio_mean
force_pred_var_to_residual_mse_ratio_mean
force_residual_rmse_to_pred_sigma_ratio_mean
force_nll_negative_frac
```

每个力轴也会输出：

```text
force_pred_var_axis_0..5
force_true_var_axis_0..5
force_pred_var_minus_true_var_axis_0..5
force_residual_mse_axis_0..5
force_target_true_var_within_horizon_axis_0..5
force_target_pred_var_minus_true_var_within_horizon_axis_0..5
```

判断方式：

```text
正常收敛:
  force_residual_mse_mean 下降
  force_pred_sigma_mean 随残差下降而下降
  force_pred_log_sigma_min_clip_frac 接近 0

疑似方差坍塌:
  force_pred_log_sigma_min_clip_frac 长时间接近 1
  force_pred_sigma_mean 非常小
  force_pred_var_minus_true_var_mean 明显为负
  force_residual_rmse_to_pred_sigma_ratio_mean 很大
  loss_force_nll 很负但 residual_mse 没有同步变小
```

当前 `log_sigma` 在模型里裁剪到 `[-5, 3]`，所以最小 `sigma = exp(-5) ~= 0.0067`。如果 `force_pred_log_sigma_min_clip_frac` 很高，说明模型在持续顶到这个下界。

## 输出目录速查

### checkpoint

LoRA 默认 7 维 state：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora/<exp_name>
```

LoRA 带力 13 维 state：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_with_force/<exp_name>
```

全量默认 7 维 state：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce/<exp_name>
```

全量带力 13 维 state：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_with_force/<exp_name>
```

LoRA force-guided：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/<exp_name>
```

每个训练 step checkpoint 目录里通常会有：

```text
params/       # 用于推理加载的模型权重
train_state/  # 训练状态, resume 需要
assets/       # norm stats 等资产
```

启动 policy server 时传给 `--policy.dir` 的是具体 step 目录，例如：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora/flexiv_pump_lora/19999
```

当前 `num_train_steps=20_000` 时，训练循环的最后一个 step index 是 `19999`，所以最终 checkpoint 通常保存在 `19999`。中间 checkpoint 按 `save_interval=1000` 保存，例如 `1000`, `2000`, ..., `19000`。

### norm stats

norm stats 在：

```text
/root/autodl-tmp/openpi/assets/<config_name>/flexiv_pump_1bottle_inputForce/norm_stats.json
```

训练保存 checkpoint 时，会把对应 norm stats 复制到 checkpoint 的 `assets/` 里面，推理时会从 checkpoint 读取。

### Hugging Face / openpi 缓存

按本文档设置后，HF / datasets 缓存写到：

```text
/root/autodl-fs/openpi_cache/hf_home
```

openpi 预训练权重缓存当前实际写到：

```text
/root/autodl-fs/openpi
```

### CSV 指标日志

训练脚本会把完整 scalar 指标写到当前实验 checkpoint 目录：

```text
/root/autodl-fs/openpi_checkpoints/<config_name>/<exp_name>/metrics.csv
```

例如当前 force-guided 命令对应：

```text
/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/flexiv_pump_lora_force_guided/metrics.csv
```

终端只会打印短摘要，例如 `loss`, `grad_norm`, `diagnostic_loss`, `loss_fm`, `force_nll`, `force_target_nll`。完整的 `force_pred_*`, `force_target_*` 方差诊断都在 CSV 里，后续可以直接用 pandas 分析：

```python
import pandas as pd

df = pd.read_csv("/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/flexiv_pump_lora_force_guided/metrics.csv")
print(df[["step", "loss", "force_nll", "force_target_nll", "force_pred_sigma_mean"]].tail())
```

### wandb

`wandb_enabled=True` 是默认值。训练脚本会使用：

```text
project = openpi
name    = --exp-name 的值
```

如果没有配置 wandb 或不想上传，训练命令加：

```bash
--no-wandb-enabled
```

## 启动策略服务

训练完成后，查看可用 checkpoint step：

```bash
ls /root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora/flexiv_pump_lora
```

选择其中一个 step 目录，例如 `<step>`，启动 policy server：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora \
  --policy.dir=/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora/flexiv_pump_lora/<step>
```

force-guided checkpoint 如果要启用 CST 引导，启动时加 `--policy.force-guidance-from-cst`：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_flexiv_pump_1bottle_inputForce_lora_force_guided \
  --policy.dir=/root/autodl-fs/openpi_checkpoints/pi0_flexiv_pump_1bottle_inputForce_lora_force_guided/flexiv_pump_lora_force_guided/<step> \
  --policy.force-guidance-from-cst
```

启用后，policy 会保存上一次 `F_phi` 对下一时刻力的预测。下一次推理收到当前 `force` 后计算：

```text
tau = sum((f_t - mu_{f,t-1})^2 / sigma_{f,t-1}^2)
lambda(tau) = 0.2 * sigmoid(1.0 * (tau - 6.0))
```

第一次推理没有上一帧力预测，因此不会引导；从第二次推理开始生效。

无论是否启用 CST 引导，force-guided policy 都会维护一个长度为 16 的滑动 `force_history`。第一次推理时用当前力重复填满窗口，之后每次推理追加当前力并丢弃最旧力。

客户端传入 observation 时字段应保持为：

```python
observation = {
    "observation/image": image,
    "observation/wrist_image": wrist_image,
    "observation/state": state,
    "prompt": "Press the pump dispenser on the bottle all the way down.",
}
```

默认 baseline 配置中 `state` 使用前 7 维。如果服务的是 `*_with_force` 或任何 `*_force_guided` checkpoint，`state` 需要传原始 13 维。

返回的 `actions` 形状是：

```text
(action_horizon, 7)
```

其中 7 维动作含义是：

```text
target_eef_pose(6), target_gripper_width(1)
```

## 快速检查

检查 dataloader 是否能正常读到数据和 norm stats：

```bash
export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-fs/openpi_cache/hf_datasets
export HF_HOME=/root/autodl-fs/openpi_cache/hf_home

uv run python - <<'PY'
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

cfg = _config.get_config("pi0_flexiv_pump_1bottle_inputForce_lora")
dl = _data_loader.create_data_loader(cfg, shuffle=False, num_batches=1, skip_norm_stats=False)
obs, actions = next(iter(dl))

print("state", obs.state.shape, obs.state.dtype)
print("actions", actions.shape, actions.dtype)
print("images", {k: v.shape for k, v in obs.images.items()})
print("prompt", obs.tokenized_prompt.shape, obs.tokenized_prompt_mask.shape)
PY
```

正常输出应类似：

```text
state (16, 32) float32
actions (16, 50, 32) float32
images {'base_0_rgb': (16, 224, 224, 3), 'left_wrist_0_rgb': (16, 224, 224, 3), 'right_wrist_0_rgb': (16, 224, 224, 3)}
prompt (16, 48) (16, 48)
```

注意这里的 `state (16, 32)` 是模型输入最终 pad 后的形状；默认配置在 pad 前只使用原始 state 的前 7 维，`*_with_force` 配置在 pad 前使用完整 13 维。

检查 force-guided batch：

```bash
export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-fs/openpi_cache/hf_datasets
export HF_HOME=/root/autodl-fs/openpi_cache/hf_home

uv run python - <<'PY'
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

cfg = _config.get_config("pi0_flexiv_pump_1bottle_inputForce_lora_force_guided")
dl = _data_loader.create_data_loader(cfg, shuffle=False, num_batches=1, skip_norm_stats=False)
obs, actions = next(iter(dl))

print("state", obs.state.shape, obs.state.dtype)
print("force", obs.force.shape, obs.force.dtype)
print("force_history", obs.force_history.shape, obs.force_history.dtype)
print("force_targets", obs.force_targets.shape, obs.force_targets.dtype)
print("force_task_target", obs.force_task_target.shape, obs.force_task_target.dtype)
print("actions", actions.shape, actions.dtype)
PY
```

正常输出应类似：

```text
state (16, 32) float32
force (16, 6) float32
force_history (16, 16, 6) float32
force_targets (16, 50, 6) float32
force_task_target (16, 6) float32
actions (16, 50, 32) float32
```
