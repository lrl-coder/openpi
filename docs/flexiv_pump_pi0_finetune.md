# flexiv_pump_1bottle_inputForce 微调 pi0_base

本文档记录当前机器上使用 openpi 微调 `pi0_base` 的运行方式。数据集已经是 LeRobot dataset 格式：

```text
/root/autodl-tmp/data/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce
```

当前已在 `src/openpi/training/config.py` 中加入四个训练配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora
pi0_flexiv_pump_1bottle_inputForce_lora_with_force
pi0_flexiv_pump_1bottle_inputForce
pi0_flexiv_pump_1bottle_inputForce_with_force
```

推荐优先使用 `pi0_flexiv_pump_1bottle_inputForce_lora`。openpi README 中给出的显存参考是：LoRA 微调约需要 22.5GB 以上显存，全量微调约需要 70GB 以上显存。

## 训练配置在哪里看

主要看这个文件：

```text
/root/autodl-tmp/openpi/src/openpi/training/config.py
```

当前 flexiv pump 相关配置在 `_CONFIGS` 列表里：

```text
pi0_flexiv_pump_1bottle_inputForce_lora            # LoRA, 默认只用 state 前 7 维
pi0_flexiv_pump_1bottle_inputForce_lora_with_force # LoRA, 使用完整 13 维 state
pi0_flexiv_pump_1bottle_inputForce                 # 全量微调, 默认只用 state 前 7 维
pi0_flexiv_pump_1bottle_inputForce_with_force      # 全量微调, 使用完整 13 维 state
```

这些配置当前显式设置了：

```text
num_train_steps = 20_000
batch_size      = 16
num_workers     = 0
weight_loader   = gs://openpi-assets/checkpoints/pi0_base/params
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

也就是说，默认每 100 step 打印/记录一次日志，每 1000 step 保存一次 checkpoint，且每 5000 step 的 checkpoint 会被保留。

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
export HF_DATASETS_CACHE=/root/autodl-tmp/openpi/.cache/huggingface/datasets
export HF_HOME=/root/autodl-tmp/openpi/.cache/huggingface
```

如果 `/root/autodl-tmp` 空间不足，改用 `/root/autodl-fs`：

```bash
cd /root/autodl-tmp/openpi

export OMP_NUM_THREADS=1
export HF_LEROBOT_HOME=/root/autodl-tmp/data/force_vla_data/data_lerobot
export HF_DATASETS_CACHE=/root/autodl-fs/openpi_cache/huggingface/datasets
export HF_HOME=/root/autodl-fs/openpi_cache/huggingface
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

默认按本文档设置后，缓存写到：

```text
/root/autodl-tmp/openpi/.cache/huggingface
```

如果切到 `/root/autodl-fs`，缓存写到：

```text
/root/autodl-fs/openpi_cache/huggingface
```

openpi 预训练权重缓存当前实际写到：

```text
/root/autodl-fs/openpi
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

客户端传入 observation 时字段应保持为：

```python
observation = {
    "observation/image": image,
    "observation/wrist_image": wrist_image,
    "observation/state": state,
    "prompt": "Press the pump dispenser on the bottle all the way down.",
}
```

默认配置中 `state` 使用前 7 维。如果服务的是 `*_with_force` checkpoint，`state` 需要传 13 维。

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
