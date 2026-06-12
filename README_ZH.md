# openpi

openpi 是一个开源的机器人模型和工具包，由 [Physical Intelligence 团队](https://www.physicalintelligence.company/) 发布。

目前，该仓库包含三种类型的模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)，基于流的视觉-语言-动作模型 (VLA)。
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)，基于 FAST 动作分词器的自回归 VLA。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)，π₀ 的升级版本，通过 [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation) 训练，以获得更好的开放世界泛化能力。在此仓库中，目前我们仅支持 π₀.₅ 的流匹配头用于训练和推理。

对于所有模型，我们提供 _基础模型_ 检查点，这些检查点已在 10k+ 小时的机器人数据上进行预训练，并提供即开即用示例或微调到自定义数据集的示例。

这是一个实验：π₀ 是为我们的机器人开发的，这些机器人不同于广泛使用的平台如 [ALOHA](https://tonyzhaozh.github.io/aloha/) 和 [DROID](https://droid-dataset.github.io/)。尽管我们乐观地认为研究人员可以将 π₀ 适配到自己的平台上进行创造性实验，但我们不期望每次尝试都成功。换句话说，π₀ 可能适用，也可能不适用，但欢迎尝试。

## 更新日志

- [2025 年 9 月] 发布了 openpi 的 PyTorch 支持。
- [2025 年 9 月] 发布了 pi05，pi0 的升级版本，具有更好的开放世界泛化能力。
- [2025 年 9 月] 为 DROID 训练增加了 [改进的空闲过滤器](examples/droid/README_train.md#data-filtering)。
- [2025 年 6 月] 添加了 [使用 `openpi` 在完整 DROID 数据集上训练 VLA 的说明](examples/droid/README_train.md)，这是 pi0-FAST-DROID 训练流程的近似开源实现。

## 系统需求

运行仓库中的模型需要 NVIDIA GPU，最低规格如下。下表假设单 GPU 使用，也可通过训练配置中的 `fsdp_devices` 使用多 GPU 并行，减少每个 GPU 的内存需求。当前训练脚本尚不支持多节点训练。

| 模式                   | 所需内存      | 示例 GPU           |
| ---------------------- | ------------ | ----------------- |
| 推理                   | > 8 GB       | RTX 4090          |
| 微调 (LoRA)            | > 22.5 GB    | RTX 4090          |
| 微调 (Full)            | > 70 GB      | A100 (80GB) / H100 |

仓库已在 Ubuntu 22.04 上测试，目前不支持其他操作系统。

## 安装

克隆仓库时，请确保更新子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 或者如果已经克隆了仓库：
git submodule update --init --recursive
````

使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。请参见 [uv 安装指南](https://docs.astral.sh/uv/getting-started/installation/) 进行设置。安装 uv 后，运行以下命令设置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 用于拉取 LeRobot 作为依赖。

**Docker**：作为 uv 安装的替代方案，我们提供 Docker 安装说明。如果系统遇到问题，可以使用 Docker 简化安装。详情见 [Docker 设置](docs/docker.md)。

## 模型检查点

### 基础模型

提供多个基础 VLA 模型检查点，已在 10k+ 小时机器人数据上预训练，可用于微调。

| 模型      | 用例 | 描述                                                                              | 检查点路径                                          |
| ------- | -- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| π₀      | 微调 | 基础 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0) 用于微调              | `gs://openpi-assets/checkpoints/pi0_base`      |
| π₀-FAST | 微调 | 基础自回归 [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast) 用于微调 | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| π₀.₅    | 微调 | 基础 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05) 用于微调           | `gs://openpi-assets/checkpoints/pi05_base`     |

### 微调模型

提供针对不同机器人平台和任务的“专家”检查点，从基础模型微调而来，可直接在目标机器人上运行。可能不适用于特定机器人，但一些检查点（尤其 DROID）在实践中泛化较好。

| 模型                  | 用例      | 描述                                                                                        | 检查点路径                                                 |
| ------------------- | ------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| π₀-FAST-DROID       | 推理      | π₀-FAST 模型在 [DROID 数据集](https://droid-dataset.github.io/) 微调，可在新场景零次执行多种简单桌面操作任务          | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| π₀-DROID            | 微调      | π₀ 模型在 [DROID 数据集](https://droid-dataset.github.io/) 微调：推理比 π₀-FAST-DROID 快，但语言跟随可能稍弱     | `gs://openpi-assets/checkpoints/pi0_droid`            |
| π₀-ALOHA-towel      | 推理      | π₀ 模型在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据微调，可在 ALOHA 平台零次折叠多种毛巾            | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| π₀-ALOHA-tupperware | 推理      | π₀ 模型在内部 ALOHA 数据微调，可从保鲜盒中取出食物                                                            | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| π₀-ALOHA-pen-uncap  | 推理      | π₀ 模型在公共 ALOHA 数据微调，可打开笔盖                                                                 | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| π₀.₅-LIBERO         | 推理      | π₀.₅ 模型为 [LIBERO](https://libero-project.github.io/datasets) 微调，达到最先进性能（详见 LIBERO README） | `gs://openpi-assets/checkpoints/pi05_libero`          |
| π₀.₅-DROID          | 推理 / 微调 | π₀.₅ 模型在 DROID 数据集微调，并结合 knowledge insulation：推理快，语言跟随良好                                  | `gs://openpi-assets/checkpoints/pi05_droid`           |

默认情况下，检查点从 `gs://openpi-assets` 自动下载并缓存到 `~/.cache/openpi`，可通过环境变量 `OPENPI_DATA_HOME` 覆盖路径。

## 运行预训练模型推理

使用预训练检查点只需几行代码（示例 π₀-FAST-DROID）：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```

可在 [示例 notebook](examples/inference.ipynb) 中测试。

提供 DROID 和 ALOHA 平台的详细推理示例：[DROID](examples/droid/README.md) / [ALOHA](examples/aloha_real/README.md)。

**远程推理**：可在不同服务器上运行模型并通过 websocket 将动作发送给机器人，详情见 [远程推理文档](docs/remote_inference.md)。

**无需机器人测试推理**：提供 [示例脚本](examples/simple_client/README.md)，生成随机观察并运行推理。

## 在自有数据上微调基础模型

以 π₀.₅ 在 LIBERO 数据集微调为例，步骤如下：

1. 转换数据为 LeRobot 数据集
2. 定义训练配置并运行训练
3. 启动策略服务器并运行推理

### 1. 转换数据为 LeRobot 数据集

示例脚本：[convert_libero_data_to_lerobot.py](examples/libero/convert_libero_data_to_lerobot.py) 可转换 LIBERO 数据，也可修改以转换自定义数据：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

若仅微调 LIBERO，可跳过此步骤。

### 2. 定义训练配置并运行训练

* [`LiberoInputs` / `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义数据映射
* [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：处理原始 LIBERO 数据
* [`TrainConfig`](src/openpi/training/config.py)：微调超参数、数据配置及权重加载器

计算训练数据归一化统计：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

启动训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

可在控制台查看日志，并保存到 `checkpoints` 目录。可使用 Weights & Biases 监控训练。`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 最大化 GPU 使用。

**注意**：支持从预训练加载状态/动作归一化统计，便于新任务微调。详见 [norm_stats.md](docs/norm_stats.md)。

### 3. 启动策略服务器并运行推理

训练完成后，启动服务器：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

服务器监听 8000 端口，可被评估脚本查询。

推荐使用 Docker 化流程处理策略服务器和评估脚本：[LIBERO README](examples/libero/README.md)。

### 更多示例

提供 ALOHA 平台的微调与推理示例：

* [ALOHA Simulator](examples/aloha_sim)
* [ALOHA Real](examples/aloha_real)
* [UR5](examples/ur5)

## PyTorch 支持

openpi 提供 π₀ 和 π₀.₅ 的 PyTorch 实现，已在 LIBERO 基准验证（推理和微调）。部分功能暂不支持：

* π₀-FAST 模型
* 混合精度训练
* FSDP 训练
* LoRA 微调
* EMA 权重

### 安装与设置

1. 更新依赖：`uv sync`
2. 确认 transformers 版本：`uv pip show transformers`
3. 应用 transformers 补丁：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

**注意**：默认 uv 链接模式下，该操作会永久影响 transformers 库，若要完全恢复需 `uv cache clean transformers`。

### JAX 模型转换为 PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### PyTorch 推理

与 JAX API 相同，只需指向转换后的检查点：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

policy = policy_config.create_trained_policy(config, checkpoint_dir)
action_chunk = policy.infer(example)["actions"]
```

### PyTorch 策略服务器

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### PyTorch 微调

1. 转换 JAX 基础模型
2. 在配置中指定 `pytorch_weight_path`
3. 启动训练，可单 GPU、多 GPU 或多节点，命令与参数详见原文

### 精度设置

**JAX**：

* 推理：大部分权重和计算为 bfloat16，部分 float32
* 训练：混合精度，权重/梯度 float32，大部分激活 bfloat16，可全 float32

**PyTorch**：

* 推理：与 JAX 一致
* 训练：支持 bfloat16 或 float32，可配置 `pytorch_training_precision`。混合精度暂不支持。

### 排错

| 问题             | 解决方案                                                               |
| -------------- | ------------------------------------------------------------------ |
| `uv sync` 依赖冲突 | 删除虚拟环境目录并重新执行 `uv sync`，确保 uv 更新至最新版本                              |
| GPU 内存不足       | 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 或使用 FSDP 多 GPU             |
| 策略服务器连接错误      | 检查服务器是否运行并监听预期端口，确认网络和防火墙设置                                        |
| 训练缺少归一化统计      | 先运行 `scripts/compute_norm_stats.py`                                |
| 数据集下载失败        | 检查网络，HuggingFace 数据集需登录                                            |
| CUDA/GPU 错误    | 确认 NVIDIA 驱动正确安装，Docker 确认 nvidia-container-toolkit 安装，系统 CUDA 可卸载 |
| 导入错误           | 确保运行 `uv sync` 安装所有依赖                                              |
| 动作维度不匹配        | 检查数据处理变换是否匹配机器人动作空间定义                                              |
| 训练发散           | 检查 `norm_stats.json` 中的 q01、q99、std，必要时手动调整                        |


