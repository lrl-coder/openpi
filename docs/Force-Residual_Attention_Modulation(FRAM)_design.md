# Force-Residual Attention Modulation (FRAM)

## 1. 核心问题

这一模块不再做“未来力目标构造”，也不再做“观测力锚定的 NLL guidance”。更合理的目标是：

> 模型先预测未来力序列；机器人执行动作后，力传感器会产生新的真实力数据。下一次推理时，我们比较已经执行时间步上的预测力和真实力。如果差距大，说明当前接触状态没有被上一轮动作预测解释，Action expert 应该更强地关注 force token。

因此，这个模块的核心不是 tracking 一个手工力参考，而是 **用已发生的预测误差调制下一轮动作生成中 force token 的注意力强度**。

这与三个已有研究脉络一致：

- action chunk policy 适合 receding-horizon 执行，每轮只执行一部分动作，再重新观测和规划 [1,2]。
- 接触丰富操作需要快速触觉/力反馈闭环，不能只依赖视觉和 open-loop chunk [3,4,5]。
- gated multimodal fusion、FiLM 和 conditional computation 说明：当某个模态更可靠或更关键时，可以用门控机制动态调节该模态对主干网络的影响 [6,7,8]。

## 2. 设计名称

建议将模块命名为：

**Force-Residual Attention Modulation (FRAM)**

这个名字比旧名更准确，也更不容易引起误解：

- **Force-Residual**：核心信号来自预测力和真实传感器力之间的时间对齐残差。
- **Attention**：模块目标是让 Action expert 在需要时更关注 force token。
- **Modulation**：残差不是选择另一条路径，而是连续调制 force token 的条件强度。

换句话说，FRAM 不是一个额外 controller，也不是一个 force target planner，而是一个基于接触预测误差的 token-level attention modulation 机制。

## 3. $F_\phi$ 预测未来力序列

在时刻 $t$，Action expert 生成动作 chunk：

$$
a_{t:t+H-1}
$$

$F_\phi$ 预测该动作 chunk 的未来力后果：

$$
F_\phi(o_t,a_{t:t+H-1})
\rightarrow
\left\{
\left(
\mu_{t+h|t},
\Sigma_{t+h|t}
\right)
\right\}_{h=1}^{H}
$$

训练时，$F_\phi$ 由真实未来力序列监督：

$$
\mathcal{L}_{F}
=
\frac{1}{H}
\sum_{h=1}^{H}
-\log
\mathcal{N}
\left(
F_{t+h};
\mu_{t+h|t},
\Sigma_{t+h|t}
\right)
$$

这里的“分布”不是说单个时间步的力标签不确定，而是为了建模部分观测、接触随机性、执行误差和传感噪声下的预测置信度。异方差输出在机器人接触建模中很有价值：均值给出预测后果，方差给出该预测在当前状态和动作下是否可靠。这个 $F_\phi$ 相当于一个 action-to-force consequence model，类似于在 action policy 旁边学习一个可解释的接触后果预测器。

这一设计与 $\pi_0$ / flow-matching action expert 的动作 chunk 生成范式兼容 [1]，也与 Diffusion Policy 中“预测动作 chunk、滚动执行”的策略形式一致 [2]。

## 4. 已执行时间步上的力残差

假设上一轮从时刻 $t$ 开始执行了 $m$ 个低层控制步。到下一次推理时刻 $t+m$，传感器已经给出真实力序列：

$$
\left\{
F^{obs}_{t+1},
F^{obs}_{t+2},
\ldots,
F^{obs}_{t+m}
\right\}
$$

这些力不是未来信息；它们已经发生，并且对应上一轮 $F_\phi$ 预测序列的前 $m$ 个时间步。因此可以计算时间对齐的预测误差：

$$
e_{t+h}
=
F^{obs}_{t+h}
-
\mu_{t+h|t},
\qquad
h=1,\ldots,m
$$

用 $F_\phi$ 的预测方差做标准化，得到一个窗口残差分数：

$$
r_{t+m}
=
\frac{1}{m}
\sum_{h=1}^{m}
e_{t+h}^{\top}
\Sigma_{t+h|t}^{-1}
e_{t+h}
$$

这个分数的含义非常清楚：上一轮动作 chunk 已经执行的部分，其真实接触响应是否符合模型预测。

- $r_{t+m}$ 小：当前视觉、语言、状态和力条件已经足够解释接触演化，Action expert 可以按常规方式生成动作。
- $r_{t+m}$ 大：真实接触响应偏离预测，说明当前接触状态对任务更关键，下一轮 Action expert 应更依赖 force token。

这正是 receding-horizon control 的自然闭环：预测、执行、观测、校正下一轮策略 [2,9]。

## 5. 残差门控为 force attention modulation

残差不再被用来构造 $F^{ref}$，也不再作为采样时 NLL 的目标。它只被转换为一个 force attention modulation coefficient：

$$
g_{t+m}
=
g_{\max}
\frac{
r_{t+m}
}{
r_{t+m}+d_f
}
$$

其中 $d_f$ 是力/力矩维度，$g_{\max}$ 是最大 force-token attention amplification。这个形式有三个好处：

- 不需要人工阈值。$r_{t+m}$ 接近 0 时，modulation 自然接近 0。
- 不需要 residual clip。大残差只会把 modulation 饱和到 $g_{\max}$。
- 尺度可解释。$d_f$ 对应标准化高斯残差的自然维度尺度。

这个门控属于 conditional modulation：网络不是一直强行依赖 force，而是在预测-观测失配时增加 force token 对 Action expert 的条件影响。这一思想与 GMU / FiLM 中“条件信号调制模态特征”的思想一致 [6,7]，也与 conditional computation 中根据输入动态改变计算/特征贡献的思想一致 [8]。

## 6. Action expert 如何更关注 force token

当前模型中，短历史力经过 local force encoder 得到一个 force token：

$$
z^{force}_t
=
E_{force}
\left(
F_{t-L+1:t}
\right)
$$

它被插入 Action expert 的 suffix tokens，使 state/action tokens 可以通过 attention 使用该接触信息。

FRAM 在下一轮推理时不改变 VLM prefix，不改变语言，不改变动作 loss，也不额外构造力目标。它只调制 force token：

$$
\tilde{z}^{force}_t
=
\left(
1+g_t
\right)
z^{force}_t
$$

然后 Action expert 用

$$
\left[
\tilde{z}^{force}_t,
z^{state}_t,
z^{action}_t
\right]
$$

进行 flow-matching denoising。

从注意力机制角度看，放大 force token 会同时增强它作为 key/value 的可见性，使后续 state/action tokens 更容易从力条件中读取接触信息。这比“用外部梯度改动作”更自然，因为它仍然让 Action expert 自己决定如何根据 force token 调整动作。

这一设计与 Visuo-Tactile Transformer 中利用触觉 token / cross-modal attention 改善操作表征一致 [4]，也与 Adaptive Visuo-Tactile Fusion 中根据 force prediction 动态调整视觉-触觉注意力的思想一致 [5]。区别是：我们的 gate 来自上一轮动作后果的 prediction error，而不是静态地融合传感器特征。

## 7. 为什么它比 NLL guidance 更优雅

旧设计的问题是：一旦我们把当前真实力写进采样 NLL，就很容易被质疑为“是不是假设下一步力等于当前力”或“是不是构造了一个虚拟力目标”。新设计避免了这一点。

FRAM 只回答一个问题：

$$
\text{上一轮模型预测的接触后果可靠吗？}
$$

如果可靠，force token 正常参与；如果不可靠，说明接触反馈比视觉/语言先验更重要，于是 Action expert 对 force token 的依赖增强。

因此，它不是 controller，也不是目标力规划器，而是一个 **closed-loop force-attention modulator**。

这个定位对审稿人更清楚：

- $F_\phi$ 的训练目标是未来力预测。
- 推理时真实力只用于评估已经执行部分的预测误差。
- 预测误差只调节下一轮 force token 的条件强度。
- 动作仍由原始 Action expert / flow policy 生成。

## 8. 与论文三个模块的关系

整个方法可以写成三段式：

1. **Local force conditioning**  
   短历史力编码为 force token，作为 Action expert 的接触状态条件。视觉-触觉/力 token 融合在机器人操作中已有充分依据 [4,5]。

2. **$F_\phi$ force consequence prediction**
   $F_\phi$ 预测动作 chunk 的未来力序列，训练时由真实未来力监督。这让模型知道“动作会造成什么接触后果”。

3. **FRAM closed-loop force attention modulation**
   推理时，将已经执行时间步的真实力与上一轮预测力对齐比较。若残差大，则放大下一轮 force token，使 Action expert 在 denoising 中更关注接触反馈。

链条如下：

$$
\text{action chunk}
\rightarrow
F_\phi \text{ predicts future force sequence}
\rightarrow
\text{execute first }m\text{ steps}
\rightarrow
\text{compare predicted vs observed force}
\rightarrow
\text{modulate Action expert attention to force}
$$

## 9. 推荐论文表述

英文表述：

> Force-Residual Attention Modulation (FRAM) is a causal token-level modulation mechanism for contact-rich action generation. At each inference cycle, the policy predicts both an action chunk and its future force consequences. After executing the first few steps, the newly observed force measurements are aligned with the corresponding predicted force steps to compute a normalized force-residual score. Rather than constructing a future force target, FRAM converts this score into an attention modulation coefficient that amplifies the local force token in the next Action Expert forward pass. The policy therefore attends more strongly to force feedback only when the previously predicted contact consequences disagree with the real sensor feedback.

中文表述：

> Force-Residual Attention Modulation (FRAM) 是一个面向接触丰富动作生成的因果 token-level modulation 机制。每次推理时，策略同时预测动作 chunk 及其未来力后果；执行前若干步后，新获得的真实力与对应预测力时间对齐，并计算标准化力残差。FRAM 不构造未来力目标，而是把该残差转换为 attention modulation coefficient，在下一次 Action expert 前向中放大 local force token。这样，只有当上一轮预测接触后果与真实传感反馈不一致时，策略才会更强地关注力反馈。

## 10. 代码对应关系

实现上应遵循：

- `policy.py` 缓存上一轮 `predict_force(observation, actions)` 得到的完整未来力序列。
- 下一轮推理时，用已经观测到的真实力与上一轮预测力前缀做时间对齐比较。
- 如果部署侧只能提供当前力，则退化为 $m=1$ 的一拍闭环。
- 如果部署侧提供最近执行窗口的 `force_history_local`，则可用窗口内的多个已执行步计算平均标准化残差。
- `policy.py` 将标准化残差分数作为 `force_prediction_error` 传给 `sample_actions`。
- `pi0.py` 将 `force_prediction_error` 映射为 `force_attention_modulation`，并在 `embed_suffix` 中调制 local force token：$\tilde{z}^{force}=(1+g)z^{force}$。
- 新配置字段使用 `force_attention_modulation_max` 表示最大 FRAM 强度；旧的 `force_guidance_lambda_max` 保留为兼容别名。
- 推理启动时优先使用 `force_attention_modulation_from_residual`；旧的 `force_guidance_from_residual` 和 `force_guidance_from_cst` 保留为兼容别名。
- 旧的 `force_target_mu/log_sigma` 采样 NLL guidance 不再作为默认 FRAM 路径。

## 11. 最终公式摘要

上一轮预测：

$$
F_\phi(o_t,a_{t:t+H-1})
\rightarrow
\left\{
\mu_{t+h|t},
\Sigma_{t+h|t}
\right\}_{h=1}^{H}
$$

已执行窗口残差：

$$
r_{t+m}
=
\frac{1}{m}
\sum_{h=1}^{m}
\left(
F^{obs}_{t+h}
-
\mu_{t+h|t}
\right)^{\top}
\Sigma_{t+h|t}^{-1}
\left(
F^{obs}_{t+h}
-
\mu_{t+h|t}
\right)
$$

force attention modulation：

$$
g_{t+m}
=
g_{\max}
\frac{r_{t+m}}{r_{t+m}+d_f}
$$

Action expert 条件调制：

$$
\tilde{z}^{force}_{t+m}
=
\left(
1+g_{t+m}
\right)
z^{force}_{t+m}
$$

一句话总结：

> FRAM 不再把残差变成力目标，而是把残差变成 Action expert 的 force-attention modulation signal。

## 12. 参考文献

[1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky. **$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control**. arXiv, 2024. [arXiv](https://arxiv.org/abs/2410.24164), [project](https://www.pi.website/research/pi0)

[2] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**. RSS, 2023. [project](https://diffusion-policy.cs.columbia.edu/), [RSS](https://roboticsconference.org/2023/program/papers/026/)

[3] Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, Cewu Lu. **Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation**. RSS, 2025. [project](https://reactive-diffusion-policy.github.io/), [arXiv](https://arxiv.org/abs/2503.02881)

[4] Yizhou Chen, Andrea Sipos, Mark Van der Merwe, Nima Fazeli. **Visuo-Tactile Transformers for Manipulation**. CoRL, 2022. [arXiv](https://arxiv.org/abs/2210.00121), [PMLR](https://proceedings.mlr.press/v205/chen23d.html)

[5] Jinzhou Li, Tianhao Wu, Jiyao Zhang, Zeyuan Chen, Haotian Jin, Mingdong Wu, Yujun Shen, Yaodong Yang, Hao Dong. **Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation**. arXiv, 2025. [arXiv](https://arxiv.org/abs/2505.13982), [project](https://adaptac-dex.github.io/)

[6] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville. **FiLM: Visual Reasoning with a General Conditioning Layer**. AAAI, 2018. [arXiv](https://arxiv.org/abs/1709.07871)

[7] John Arevalo, Thamar Solorio, Manuel Montes-y-Gomez, Fabio A. Gonzalez. **Gated Multimodal Units for Information Fusion**. ICLR Workshop, 2017. [OpenReview](https://openreview.net/forum?id=S12_nquOe), [arXiv PDF](https://arxiv.org/pdf/1702.01992)

[8] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. ICLR, 2017. [OpenReview](https://openreview.net/forum?id=B1ckMDqlg), [arXiv](https://arxiv.org/abs/1701.06538)

[9] Armand Jordana, Sebastien Kleff, Justin Carpentier, Nicolas Mansard, Ludovic Righetti. **Force Feedback Model-Predictive Control via Online Estimation**. ICRA, 2024. [HAL](https://hal.science/hal-04564888v1/document)
