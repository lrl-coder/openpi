# 基于 pi0 的力觉增强 VLA 方案可行性与创新性分析

本文档分析当前 force-guided pi0 方案的技术可行性与研究创新性。分析对象是通用接触操作场景下的力觉增强 Vision-Language-Action 模型，而非某一个具体任务。当前代码实现以 `pi0_flexiv_pump_1bottle_inputForce_lora_force_guided` 为锚点：模型基于 `pi0_base`，采用 LoRA 微调，在 pi0 的视觉-语言-动作生成框架中加入力语义监督、局部力条件和真实力闭环引导。

## 1. 方案定位

pi0 的核心思想是将预训练视觉语言模型的语义理解能力与连续机器人动作生成结合起来，通过 flow matching 生成动作序列。该范式为通用机器人策略提供了清晰结构：VLM prefix 负责图像和语言上下文建模，action expert suffix 负责条件动作生成。与 RT-2、Octo 等 VLA / generalist robot policy 工作一致，pi0 代表了从单任务视觉模仿学习走向通用机器人基础模型的趋势 [[1]](https://arxiv.org/abs/2410.24164) [[2]](https://proceedings.mlr.press/v229/zitkovich23a.html) [[3]](https://roboticsconference.org/2024/program/papers/90/)。

但在接触丰富的操作任务中，仅依赖图像、语言和低维状态存在明显不足。接触力、滑移、碰撞、阻抗变化和受力异常往往不能从单帧视觉中直接可靠推断。力/力矩传感器提供了视觉之外的物理交互信息，对接触稳定性、动作安全性和闭环修正具有直接价值。多模态触觉/力觉研究已经证明，视觉和触觉/力觉之间存在可学习的互补关系，接触模态可以显著增强接触操作中的表征与控制能力 [[10]](https://arxiv.org/abs/2211.12498) [[11]](https://proceedings.mlr.press/v205/li23c.html) [[12]](https://arxiv.org/abs/1810.10191)。

因此，本方案的目标不是简单地把力信号拼接到状态向量中，而是在 pi0 的原有结构内引入三类互补机制：

```text
1. semantic-force query：用真实历史力监督 VLM 侧接触语义表征。
2. local force conditioning：用短历史力条件化 action expert。
3. Causal Force Residual Guidance：用预测力残差进行推理期采样引导。
```

这种设计保持了 pi0 的主干能力，同时将力信号拆分为训练语义监督、动作动态建模和在线闭环反馈三个角色。

## 2. 当前实现概述

当前实现对应配置：

```text
pi0_flexiv_pump_1bottle_inputForce_lora_force_guided
```

关键设置：

```text
base checkpoint: pi0_base
fine-tuning: LoRA
force_guidance: True
force_loss_weight: 0.05
force_target_loss_weight: 0.01  # 当前作为 semantic-force alignment 权重
force_guidance_lambda_max: 0.2
```

模型输入中，常规 pi0 state token 仍只使用低维机器人状态；历史力不进入 VLM prefix。力相关字段被拆成：

```text
force_history_global: 64 x 6  # 长历史力，用作训练期 force teacher 输入
force_history_local: 16 x 6   # 短历史力，用作 action suffix 条件
force_targets: 50 x 6         # 未来动作 chunk 对应的力监督
force: 6                      # 当前真实力，用于在线力预测残差计算
```

总损失为：

```text
L = L_FM + lambda_phi * L_F_phi + lambda_sem * L_sem
```

其中：

```text
L_FM: pi0 原始 flow matching action loss
L_F_phi: action hidden state 到未来力分布的 Gaussian NLL
L_sem: semantic-force query feature 与历史力 teacher feature 的 cosine alignment
```

当前实现明确不使用未来力预测头作为语义力目标。语义力目标只保留主对齐损失。

## 3. 方法结构

### 3.1 pi0 主干

pi0 的 VLM prefix 处理图像和语言，action expert suffix 处理状态、噪声动作和时间步，并通过 flow matching 学习动作速度场。Flow Matching 本身是成熟的生成式建模框架，能够以回归向量场的形式学习从噪声到数据的连续变换 [[5]](https://openreview.net/forum?id=PqvMRDCJT9t)。Diffusion Policy 也从机器人策略角度证明，迭代式生成模型适合高维、连续、多模态动作分布 [[4]](https://roboticsconference.org/2023/program/papers/026/)。因此，在 pi0 的动作生成过程中加入可微引导项具有方法基础。

### 3.2 Semantic-Force Query Path

本方案不再将 `force_history_global` patch 化后送入 VLM prefix，而是在 VLM prefix 中加入一个可学习的 semantic-force query token：

```text
[q_force, image_tokens, language_tokens] -> VLM prefix
```

该 query token 与图像和语言 token 双向注意力交互。经过 VLM 后，取 query token 对应的输出 hidden state：

```text
h_query -> projection head -> z_query
```

`z_query` 表示模型从视觉语言上下文中抽取到的接触/力语义特征。该设计避免了简单 prefix pooling 的语义混杂，也避免了训练时让 VLM 直接看到历史力造成的 force-token leakage。

learnable query 的可行性可以从多个顶会工作中得到支持。BERT 使用特殊 token 聚合序列级语义 [[8]](https://arxiv.org/abs/1810.04805)；DETR 使用 learned object queries 从图像上下文中查询目标 [[6]](https://arxiv.org/abs/2005.12872)；Perceiver 使用 latent bottleneck / latent queries 从高维多模态输入中提取紧凑表征 [[7]](https://proceedings.mlr.press/v139/jaegle21a.html)；Flamingo 通过 Perceiver Resampler 压缩视觉特征并接入语言模型 [[9]](https://arxiv.org/abs/2204.14198)。这些工作共同说明：用少量可学习 token 在 Transformer 中读取特定语义，是成熟且有效的结构范式。

### 3.3 Force Teacher Path

训练阶段，真实长历史力 `force_history_global` 进入轻量 force teacher encoder：

```text
force_history_global: B x 64 x 6
-> causal / dilated Conv1D encoder
-> temporal pooling
-> projection head
-> z_force
```

`z_force` 表示由真实力序列编码出的接触动力学特征。它不进入 VLM prefix，也不在推理时作为 VLM 输入。语义力目标为：

```text
L_sem = 1 - cosine(z_query, z_force)
```

这个目标的角色是 privileged supervision：训练时真实力历史可见，用它监督 VLM 从图像语言中学习接触语义；推理时 VLM 只需要 query token、图像和语言即可产生 force-aware semantic feature。该思想与 modality hallucination 和 cross-modal distillation 一致：训练时利用额外模态提供监督，测试时不依赖该额外模态 [[13]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Hoffman_Learning_With_Side_CVPR_2016_paper.html) [[14]](https://arxiv.org/abs/1507.00448)。

### 3.4 Local Force Conditioning Path

短历史力 `force_history_local` 进入局部力编码器，生成一个 action suffix 条件 token：

```text
force_history_local: B x 16 x 6
-> causal dilated Conv1D
-> force condition token
-> action expert suffix
```

这一路径面向短时接触动态，作用对象是动作生成，而不是 VLM 语义。它与 semantic-force query 解耦：前者服务于 action expert 的局部控制条件，后者服务于 VLM 侧接触语义学习。

### 3.5 Action-Level Force Predictor F_phi

`F_phi` 挂在 action expert 的 action-token hidden features 后面，预测动作 chunk 中每个动作对应的未来力分布：

```text
action hidden features -> F_phi -> mu_f, log_sigma_f
```

监督目标是 `force_targets: 50 x 6`，损失为 Gaussian NLL：

```text
L_F_phi = NLL(force_targets; mu_f, sigma_f)
```

`F_phi` 的作用不是生成语义力目标，而是让 action hidden state 显式承载动作-力动态关系，并为在线 CFRG 提供上一轮动作条件下的力预测分布。

### 3.6 Causal Force Residual Guidance

CFRG 用当前真实测量力和上一轮 `F_phi` 的一步预测计算因果力残差：

$$
e_t = f_t - \mu^{t-1}_0
$$

再用预测方差归一化，得到引导门控分数：

$$
r_t = e_t^\top \left(\Sigma^{t-1}_0\right)^{-1} e_t
$$

当残差较大时，guidance strength 增大：

$$
\lambda(r_t) = \lambda_{max} \frac{r_t}{r_t + d_f}
$$

当前实现不再把当前真实力直接当作整段未来力目标，也不构造完整未来力轨迹。CFRG 只使用短历史力构造近端一步参考：

$$
F^{ref}_{t+1}
=
F_t + (F_t - F_{t-1})
$$

并只约束候选动作的第一个未来力预测：

$$
F_\phi(o_t,a)_0
\approx
F^{ref}_{t+1}
$$

也就是说，semantic-force query 负责训练表征，`F_phi` 负责动作条件下的未来力预测，CFRG 负责真实力闭环反馈和采样引导，三者不混用。这样做避免了在线引导依赖 VLM 力目标，也避免了“当前力等于未来目标”的过强假设。

## 4. 可行性分析

### 4.1 架构可行性

pi0 的分层结构天然适合加入力觉扩展。VLM prefix、action expert suffix 和 flow matching head 在功能上相对清晰，因此力信号可以分别进入语义层、动作层和采样反馈层，而不需要重写主干架构。当前实现只在 `force_guidance=True` 的配置中生效，不影响普通 pi0 LoRA、with-force state baseline 或全量微调配置。

LoRA 微调进一步提高工程可行性。LoRA 已在 ICLR 2022 中证明，大模型适配可以通过低秩参数更新完成，显著减少可训练参数和显存开销 [[15]](https://openreview.net/forum?id=nZeVKeeFYf9)。本方案在 pi0 LoRA 配置上添加轻量 force module，相比重训整个 VLA 主干更实际。

结论：从模型结构和训练资源角度看，方案具备明确可行性。

### 4.2 数据可行性

本方案不需要额外人工标注。训练所需监督来自机器人轨迹中已有的力/力矩信号：

```text
current force
force history
future force sequence
```

这类信号在接触任务中通常来自 wrist F/T sensor、外力估计或关节力矩推断。它们比人工语义标签更容易自动采集。视觉-触觉/力觉自监督和跨模态学习文献也表明，配对的视觉与触觉/力觉数据可以用于学习可迁移的接触表征 [[10]](https://arxiv.org/abs/2211.12498) [[12]](https://arxiv.org/abs/1810.10191)。

结论：只要机器人系统能记录同步力信号，该方法即可复用已有轨迹数据进行训练。

### 4.3 训练可行性

训练损失设计较轻：

```text
L_sem = 1 - cosine(z_query, z_force)
```

没有引入大规模对比学习队列、复杂多正样本采样或额外 future-force semantic target head。force teacher encoder 是小型时序网络，参数量远小于 VLM 主干。`force_target_loss_weight` 默认较小，语义力对齐作为辅助正则项存在，不应主导 action learning。

同时，`F_phi` 的 Gaussian NLL 是明确的数值监督，用于学习动作条件下的力分布。NLL 中的方差项可以表达 force prediction uncertainty，比单纯 MSE 更适合接触力这种多噪声、多不确定性的目标。

结论：训练目标可微、轻量、可诊断，具备实际训练可行性。

### 4.4 推理可行性

推理时，VLM prefix 只需要：

```text
semantic-force query token
image tokens
language tokens
```

历史力不再作为 VLM 输入，因此避免了训练时依赖历史力、推理时历史力不可得或噪声较大的 mismatch。短历史力仍可用于 action suffix 的局部控制条件；当前真实力可用于 CFRG 的一步预测残差计算。这符合机器人在线控制场景：语义理解依赖视觉语言，接触修正依赖真实力反馈。

Diffusion Policy 和 Flow Matching 都说明，迭代式动作生成过程可以容纳条件引导或梯度修正 [[4]](https://roboticsconference.org/2023/program/papers/026/) [[5]](https://openreview.net/forum?id=PqvMRDCJT9t)。当前 CFRG 正是利用这一点，在采样过程中基于真实力预测残差修正动作速度场。

结论：推理链路明确，且与生成式动作策略的采样机制兼容。

### 4.5 迁移可行性

本方案不绑定具体任务。只要任务满足以下条件，即可迁移：

```text
1. 存在图像和语言条件。
2. 存在低维机器人状态和连续动作。
3. 训练数据中包含同步力/力矩信号。
4. 推理时可读取当前力，或至少可使用短历史力。
```

因此，该方案适用于泵压、插拔、旋拧、擦拭、按压、装配、开门、接触式搜索等接触丰富操作。Octo 等工作已经证明 generalist policies 可以通过微调适配新任务、新传感器和新动作空间 [[3]](https://roboticsconference.org/2024/program/papers/90/)，本方案在 pi0 上加入力觉模块，具备类似迁移潜力。

## 5. 创新性分析

### 5.1 从“力作为输入”到“力作为语义监督”

最直接的力觉增强方式是将力信号拼接到 state 中。这种做法简单，但容易把力信号当作低维数值条件处理，难以让 VLM 学到接触语义，也容易在训练和推理之间形成强依赖。

本方案将长历史力定义为训练期 privileged modality：历史力不进入 VLM prefix，而是通过 teacher encoder 生成 `z_force`，监督 semantic-force query 学习接触语义。推理时，VLM 不需要历史力输入。这一点与 modality hallucination / cross-modal distillation 的思想一致，但应用对象从 RGB-D/视觉模态迁移到 VLA 中的视觉语言-力觉接触语义 [[13]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Hoffman_Learning_With_Side_CVPR_2016_paper.html) [[14]](https://arxiv.org/abs/1507.00448)。

创新点：力信号不只是控制输入，而是用于塑造 VLM 接触语义表征的训练监督。

### 5.2 Semantic-Force Query

相比对 VLM prefix 做平均池化，learnable semantic-force query 更具有目标性。它作为专门 token 在 VLM 内部和图像、语言上下文交互，最终输出一个面向接触/力语义的表示。

该设计借鉴了 Transformer 中 query / latent token 的成熟范式，但具体用途不同：

```text
BERT: special token 聚合句级语义 [[8]](https://arxiv.org/abs/1810.04805)
DETR: object queries 查询视觉目标 [[6]](https://arxiv.org/abs/2005.12872)
Perceiver: latent queries 压缩高维多模态输入 [[7]](https://proceedings.mlr.press/v139/jaegle21a.html)
Flamingo: Perceiver Resampler 压缩视觉特征接入语言模型 [[9]](https://arxiv.org/abs/2204.14198)
本方案: semantic-force query 查询图像语言上下文中的接触语义
```

创新点：将 learnable query token 从目标检测/多模态压缩推广到 VLA 接触语义建模中，使语义力特征成为显式可学习对象。

### 5.3 语义表征、动作力预测和在线引导解耦

本方案没有用单一 force head 承担所有功能，而是拆成三个模块：

```text
semantic-force query: 学习图像语言中的接触语义
F_phi: 学习动作 hidden state 到未来力分布的映射
CFRG: 使用因果力预测残差引导采样
```

这种解耦使每条路径的监督和功能更清晰。semantic-force query 不需要直接预测未来力数值；`F_phi` 不需要承担 VLM 语义解释；CFRG 不依赖 VLM 生成的力目标，而使用真实力反馈与上一轮同时间步预测之间的残差来门控采样时的力一致性能量。

创新点：把“力语义学习”“动作-力动力学建模”“真实力闭环修正”分离，避免常见多模态融合中所有信息混在一个条件向量里的问题。

### 5.4 因果力残差引导

旧思路中，VLM semantic target head 预测 `mu*, sigma*`，再将其用于引导动作采样。但图像语言无法唯一确定真实接触力，尤其在接触刚度、摩擦、物体状态和传感器偏置变化时，VLM 预测目标可能不稳定。

当前方案改为：CFRG 使用上一轮一步力预测和当前真实测量力之间的残差作为闭环反馈，并用该残差门控下一轮采样中的 observed-force anchor。这样推理闭环更短，物理依据更强：

$$
f_t - \mu^{t-1}_0
\rightarrow
r_t
\rightarrow
\lambda_t
\rightarrow
\text{sensor-anchored force consistency}
\rightarrow
\text{guided sampling}
$$

这与机器人控制中依赖真实传感反馈进行闭环修正的原则一致，也保留了 pi0 / flow matching 动作生成的连续优化特性。

创新点：将真实力反馈转化为因果力预测残差，并用该残差门控 diffusion 采样中的观测力一致性项，而不是依赖 VLM 预测的虚拟力目标或手工构造的未来力轨迹。

### 5.5 低侵入式 pi0 增强

本方案保留 pi0 的核心结构：

```text
VLM prefix
action expert suffix
flow matching action generation
LoRA fine-tuning
```

新增模块只在 force-guided 配置中启用，不破坏 baseline。相比训练一个独立 force controller 或重构 VLA 架构，这种低侵入式设计更利于工程复现和后续迁移。

创新点：在不改变 pi0 基本范式的前提下，将力语义、力预测和真实力引导嵌入 VLA 框架。

## 6. 与已有方法的关系

### 6.1 相比普通 VLA

RT-2、Octo 和 pi0 说明，视觉语言预训练和机器人动作学习结合可以提升泛化能力 [[1]](https://arxiv.org/abs/2410.24164) [[2]](https://proceedings.mlr.press/v229/zitkovich23a.html) [[3]](https://roboticsconference.org/2024/program/papers/90/)。但这些通用 VLA 工作主要关注视觉、语言、状态和动作，并未系统处理真实力反馈在接触操作中的作用。

本方案补足的是接触操作中的物理反馈层：既让模型从力历史中学习接触语义，又在推理时使用真实力闭环引导。

### 6.2 相比直接力输入 baseline

直接把 force 拼到 state 中可以作为强 baseline，但其表达层次较低。本方案区分：

```text
长历史力 -> 语义监督
短历史力 -> 局部动作条件
当前真实力 -> 推理闭环引导
未来力序列 -> F_phi 动作力预测监督
```

因此，本方案不是简单增加传感器维度，而是将力信号按时间尺度和功能角色分配到不同模块。

### 6.3 相比传统多模态融合

传统多模态融合通常将视觉、触觉、音频等模态同时输入一个融合网络 [[11]](https://proceedings.mlr.press/v205/li23c.html)。这种方式有效，但在 VLA 大模型中可能带来推理依赖和训练-推理不一致。

本方案采用 privileged force supervision：VLM 不直接消费历史力，而是通过对齐目标学习 force-aware semantic representation。这个设计更适合基于大 VLM 的机器人策略，因为它减少了对额外模态输入的强耦合。

## 7. 风险与边界

### 7.1 可观测性边界

图像和语言不能唯一决定真实力。semantic-force query 不应被描述为“恢复真实力历史”或“精确预测力曲线”。更准确的表述是：

```text
semantic-force query learns a contact-aware semantic representation grounded by force histories during training.
```

### 7.2 数据同步风险

力信号与图像、动作必须时间对齐。如果力传感器延迟、滤波或重采样处理不一致，teacher feature 可能监督错误的视觉语言上下文。需要检查传感器同步、窗口选择和归一化统计。

### 7.3 小数据风险

小规模数据下，force teacher encoder 可能过拟合特定接触模式。应控制 `lambda_sem`，并通过 validation 指标观察：

```text
loss_force_semantic_align
force_semantic_cosine_mean
loss_fm
success rate
```

如果 semantic alignment 提升但 action loss 或成功率恶化，说明辅助目标过强。

### 7.4 CFRG 稳定性风险

CFRG guidance 强度过大可能扰动原始 action distribution。需要对以下参数做消融：

```text
force_guidance_lambda_max
```

同时应监控动作平滑性、峰值力和力波动，避免引导项造成控制震荡。

## 8. 实验与消融建议

### 8.1 主对比

建议至少包含：

```text
1. pi0 LoRA baseline
2. direct force-in-state baseline
3. force-guided without semantic-force alignment
4. force-guided with semantic-force query
5. force-guided with CFRG
```

### 8.2 模块消融

建议消融：

```text
1. remove semantic-force query
2. set lambda_sem = 0
3. remove force teacher encoder
4. remove local force token
5. remove F_phi
6. disable CFRG
7. current-force target vs one-step force-trend reference
```

### 8.3 指标

建议报告：

```text
task success rate
flow matching loss
force prediction NLL
semantic-force cosine alignment
peak force
force variance / force smoothness
action smoothness
inference latency
```

对论文而言，成功率只能说明最终效果，力曲线和动作平滑性更能说明 force-guided 机制是否真的改善了接触质量。

## 9. 预期贡献表述

可以将本文方法贡献概括为三点：

```text
1. We propose a force-guided extension of pi0 that incorporates force signals through semantic supervision, local action conditioning, and real-force-guided sampling.

2. We introduce a learnable semantic-force query that interacts with visual-language tokens and is aligned with force-history representations during training, enabling force-aware semantic grounding without feeding force histories into the VLM prefix.

3. We design Causal Force Residual Guidance (CFRG), which converts the one-step force prediction residual into an adaptive energy guidance term for flow-matching action sampling.
```

中文表述可写为：

```text
本文提出一种基于 pi0 的力觉增强 VLA 框架。该框架不简单拼接力输入，而是将历史力作为训练期语义监督，将短历史力作为局部动作条件，并在推理阶段使用因果力预测残差进行 CFRG 引导。通过 semantic-force query、动作级力预测 F_phi 和真实力闭环采样三者解耦，模型能够在保持 pi0 视觉语言动作生成能力的同时，提高接触操作中的物理感知和动作稳定性。
```

## 10. 总结

本方案具备明确可行性：pi0 的 prefix/suffix 分层结构为力觉模块提供了低侵入式接入点；LoRA 降低了微调成本；力信号可从机器人轨迹中自动获得；Flow Matching 的采样过程允许加入可微引导；触觉/力觉多模态研究也支持力信号对接触操作的重要性。

本方案的创新性在于：不把力信号仅作为状态拼接，而是将其拆分为语义监督、动作力预测和真实力闭环引导；不让历史力直接进入 VLM prefix，而是通过 learnable semantic-force query 学习接触语义；不依赖 VLM 预测在线力目标，而使用一步力预测残差进行 CFRG guidance。这使得方法既继承 pi0 的 VLA 泛化能力，又补足接触任务中视觉不可观测的物理反馈。

## 参考文献

[1] Black, K. et al. "pi0: A Vision-Language-Action Flow Model for General Robot Control." arXiv, 2024.  
https://arxiv.org/abs/2410.24164

[2] Zitkovich, B. et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." Conference on Robot Learning, 2023.  
https://proceedings.mlr.press/v229/zitkovich23a.html

[3] Ghosh, D. et al. "Octo: An Open-Source Generalist Robot Policy." Robotics: Science and Systems, 2024.  
https://roboticsconference.org/2024/program/papers/90/

[4] Chi, C. et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." Robotics: Science and Systems, 2023.  
https://roboticsconference.org/2023/program/papers/026/

[5] Lipman, Y. et al. "Flow Matching for Generative Modeling." International Conference on Learning Representations, 2023.  
https://openreview.net/forum?id=PqvMRDCJT9t

[6] Carion, N. et al. "End-to-End Object Detection with Transformers." European Conference on Computer Vision, 2020.  
https://arxiv.org/abs/2005.12872

[7] Jaegle, A. et al. "Perceiver: General Perception with Iterative Attention." International Conference on Machine Learning, 2021.  
https://proceedings.mlr.press/v139/jaegle21a.html

[8] Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL, 2019.  
https://arxiv.org/abs/1810.04805

[9] Alayrac, J.-B. et al. "Flamingo: a Visual Language Model for Few-Shot Learning." Advances in Neural Information Processing Systems, 2022.  
https://arxiv.org/abs/2204.14198

[10] Yang, F. et al. "Touch and Go: Learning from Human-Collected Vision and Touch." Advances in Neural Information Processing Systems, 2022.  
https://arxiv.org/abs/2211.12498

[11] Li, H. et al. "See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation." Conference on Robot Learning, 2023.  
https://proceedings.mlr.press/v205/li23c.html

[12] Lee, M. A. et al. "Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks." ICRA, 2019.  
https://arxiv.org/abs/1810.10191

[13] Hoffman, J., Gupta, S., and Darrell, T. "Learning with Side Information through Modality Hallucination." IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Hoffman_Learning_With_Side_CVPR_2016_paper.html

[14] Gupta, S., Hoffman, J., and Malik, J. "Cross Modal Distillation for Supervision Transfer." IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
https://arxiv.org/abs/1507.00448

[15] Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." International Conference on Learning Representations, 2022.  
https://openreview.net/forum?id=nZeVKeeFYf9
