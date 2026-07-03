# 如何表述 $F_\phi$ 的逐时间步力预测分布

## 1. 代码事实

当前实现中，$F_\phi$ 挂在 action expert 的 action-token hidden features 后面。对于一个长度为 $H$ 的动作 chunk，action expert 会输出 $H$ 个 action-token hidden states：

$$
h_{t:t+H-1}
=
\operatorname{ActionExpert}
\left(
o_t,
F_{t-L:t},
a_{t:t+H-1}
\right)
$$

$F_\phi$ 对每个 action-token hidden state 独立解码一组 6D force/torque 的高斯参数：

$$
F_\phi(h_{t+h})
=
\left(
\mu_{t+h+1},
\log \sigma_{t+h+1}
\right),
\quad h = 0,\dots,H-1
$$

其中：

- $\mu_{t+h+1} \in \mathbb{R}^6$
- $\sigma_{t+h+1} \in \mathbb{R}^6$
- 6 个维度对应 force/torque axes
- 训练标签是数据中的未来力序列 $F_{t+1:t+H}$

所以，严格按代码看，$F_\phi$ **确实是在每个 horizon step 输出一个力预测分布的参数**。

## 2. 为什么单个时间步也可以建模成分布

审稿人可能会问：一个时间步的力不是一个确定数值吗？为什么要预测分布？

这个问题的关键在于：训练数据中每条轨迹的某个时间步确实只有一个观测值，但模型在给定有限观测 $o_t$ 和候选动作 $a_{t:t+H-1}$ 时，并不能确定真实接触动力学的全部隐变量。因此我们预测的不是“标签自身的随机性”，而是 **条件预测不确定性**。

更准确地说，$F_\phi$ 建模的是：

$$
q_\phi
\left(
F_{t+1:t+H}
\mid
o_t,
a_{t:t+H-1}
\right)
$$

这里的分布 $q_\phi$ 是模型的 predictive distribution，而不是说同一个数据样本的标签有多个真实值。

在接触操作中，即使 $o_t$ 和 $a_{t:t+H-1}$ 给定，未来力仍会受到未观测因素影响，例如：

- 接触点微小偏移
- 物体刚度和形变
- 摩擦系数
- 传感器噪声和滤波延迟
- 夹持状态
- 液体、软体或机构内部状态
- 数据集中不同 episode 的隐含环境差异

这些隐变量不会完整出现在视觉、语言和低维状态中。因此，对单步未来力使用条件高斯预测是合理的。

## 3. 不应声称它是真实完整物理分布

论文里不能把 $F_\phi$ 说成“恢复了真实未来力分布”。这会过强，因为数据中没有对同一个状态-动作对重复采样，无法直接监督完整物理分布。

更稳妥的表述是：

> $F_\phi$ predicts a factorized heteroscedastic Gaussian approximation to the action-conditioned future force.

中文可以写为：

> $F_\phi$ 预测动作条件未来力的因子化异方差高斯近似。

这里有三个关键词：

- **action-conditioned**：力后果依赖动作，不是单纯观测预测。
- **heteroscedastic**：不同样本、不同时间步、不同力轴的不确定性不同。
- **factorized Gaussian approximation**：这是轻量近似，不是完整联合分布。

## 4. 推荐数学表述

给定当前观测 $o_t$、短历史力 $F_{t-L:t}$ 和候选动作 chunk $a_{t:t+H-1}$，action expert 产生动作隐状态：

$$
h_{t:t+H-1}
=
g_\theta
\left(
o_t,
F_{t-L:t},
a_{t:t+H-1}
\right)
$$

力预测头定义为：

$$
\left(
\mu_{t+1:t+H},
\log \sigma_{t+1:t+H}
\right)
=
F_\phi
\left(
h_{t:t+H-1}
\right)
$$

我们使用一个因子化对角高斯作为未来力的预测近似：

$$
q_\phi
\left(
F_{t+1:t+H}
\mid
o_t,
F_{t-L:t},
a_{t:t+H-1}
\right)
=
\prod_{h=1}^{H}
\mathcal{N}
\left(
F_{t+h};
\mu_{t+h},
\operatorname{diag}
\left(
\sigma_{t+h}^{2}
\right)
\right)
$$

训练目标是负对数似然：

$$
\mathcal{L}_{force}
=
-\sum_{h=1}^{H}
\log
\mathcal{N}
\left(
F_{t+h}^{gt};
\mu_{t+h},
\operatorname{diag}
\left(
\sigma_{t+h}^{2}
\right)
\right)
$$

展开后为：

$$
\mathcal{L}_{force}
=
\frac{1}{2}
\sum_{h=1}^{H}
\sum_{j=1}^{6}
\left[
\frac{
\left(
F_{t+h,j}^{gt} - \mu_{t+h,j}
\right)^2
}{
\sigma_{t+h,j}^{2}
}
+ 2 \log \sigma_{t+h,j}
+ \log 2\pi
\right]
$$

## 5. 为什么不是直接 MSE

如果只用 MSE：

$$
\mathcal{L}_{MSE}
=
\sum_h
\left\|
F_{t+h}^{gt} - \mu_{t+h}
\right\|_2^2
$$

模型会被迫对所有样本、所有阶段、所有轴使用同样的误差尺度。但接触力不是这样的。

例如：

- 自由空间运动阶段，力接近零且方差小。
- 接触建立阶段，力变化快且不确定性大。
- 稳定按压阶段，法向力可预测但切向力可能受摩擦影响。
- 释放或滑移阶段，力变化方向和幅值都更难预测。

异方差高斯 NLL 允许模型表达：

$$
\sigma_{t+h,j}
=
\sigma_\phi
\left(
o_t,
F_{t-L:t},
a_{t:t+H-1},
h,
j
\right)
$$

也就是说，不确定性可以随时间步、力轴和接触状态变化。

这对 CFRG 尤其重要，因为 CFRG 使用预测方差归一化一步力残差：

$$
r_t
=
\left(
F_t - \mu^{t-1}_{0}
\right)^\top
\left(
\Sigma^{t-1}_{0}
\right)^{-1}
\left(
F_t - \mu^{t-1}_{0}
\right)
$$

如果没有 $\Sigma$，所有力轴和所有接触阶段都会被同等对待，容易让噪声轴或高不确定阶段产生过强引导。

## 6. 与 Local Force Conditioning 的关系

Local force conditioning 和 $F_\phi$ 不是两个孤立模块。

短历史力 $F_{t-L:t}$ 先经过 local force encoder，形成 action expert 的条件 token：

$$
z_t^{local}
=
\operatorname{LocalForceEncoder}
\left(
F_{t-L:t}
\right)
$$

该 token 被注入 action expert：

$$
h_{t:t+H-1}
=
g_\theta
\left(
o_t,
z_t^{local},
a_{t:t+H-1}
\right)
$$

因此，$F_\phi$ 的预测不是只看动作，也不是只看当前图像，而是在 action hidden state 中同时读取：

- 视觉语言上下文
- 当前低维状态
- 候选动作 chunk
- 短历史力编码出的局部接触状态

这使得 $F_\phi$ 可以建模动作在当前接触状态下的力后果。换句话说，local force conditioning 提供了“当前接触状态”，$F_\phi$ 则学习“在该接触状态下执行某个动作会产生怎样的未来力”。

## 7. 与 CFRG 的关系

CFRG 需要一个可微的力后果模型。$F_\phi$ 正好承担这个角色。

在推理时，上一轮预测给出：

$$
\left(
\mu^{t-1}_{0:H-1},
\Sigma^{t-1}_{0:H-1}
\right)
$$

当前真实力 $F_t$ 到来后，计算一步预测残差：

$$
e_t
=
F_t - \mu^{t-1}_{0}
$$

并得到标准化残差：

$$
r_t
=
e_t^\top
\left(
\Sigma^{t-1}_{0}
\right)^{-1}
e_t
$$

这个 $r_t$ 用于控制引导强度；CFRG 不再构造整段未来力参考，而是用短历史力得到近端一步参考：

$$
F^{ref}_{t+1}
=
F_t + (F_t - F_{t-1})
$$

然后采样阶段只用当前候选动作的 $F_\phi$ 第一个未来力预测与 $F^{ref}_{t+1}$ 构造 energy。这里的关键是：$F_\phi$ 不仅在训练中提供辅助监督，也在推理中提供可微的 action-to-force critic。

## 8. 审稿人可能质疑与回应

### 8.1 质疑：单个时间步的力是确定值，为什么预测分布？

回应：

> We do not assume that a single recorded force label is itself stochastic. Instead, the Gaussian parameterization represents the model's conditional predictive uncertainty under partial observability and contact variability. Similar heteroscedastic likelihoods are commonly used when a deterministic observation is modeled with input-dependent uncertainty.

中文：

> 我们并不假设单条记录中的力标签本身是随机的。高斯参数化表示的是部分可观测条件下模型对未来力的条件预测不确定性，而不是同一标签存在多个真实值。

### 8.2 质疑：为什么每个 horizon step 独立建模？

回应：

> We use a factorized diagonal Gaussian for computational efficiency and stable integration with action diffusion guidance. The temporal dependency is not ignored by the model entirely: the action expert hidden states are produced jointly over the action chunk and condition on the shared observation and local force history. The factorization only applies to the likelihood head, not to the representation.

中文：

> 我们采用因子化对角高斯是为了计算效率和采样引导稳定性。时间相关性并没有完全被忽略，因为 action expert 是在共享观测、短历史力和整段动作 chunk 条件下联合产生 action-token hidden states。因子化只发生在 likelihood head，而不是整个模型表征层。

### 8.3 质疑：$\sigma$ 会不会只是 loss attenuation？

回应：

> This is a valid risk for heteroscedastic NLL. We therefore clip log-variance, monitor calibration metrics, and use the predicted variance only as a relative confidence signal for residual-normalized guidance rather than claiming calibrated physical noise.

中文：

> 这是异方差 NLL 的常见风险。因此实现中会裁剪 log-variance，并监控校准指标。我们只将预测方差作为残差归一化引导中的相对置信度信号，而不声称它是严格校准的物理噪声。

## 9. 论文推荐表述

可以在方法部分这样写：

> We introduce an action-conditioned force consequence head $F_\phi$ on top of the action expert hidden states. Given the current observation, a short causal force history, and an action chunk, $F_\phi$ predicts a factorized heteroscedastic Gaussian approximation to the future force/torque sequence. This predictive distribution is not intended to recover the full physical stochasticity of contact. Instead, it provides a lightweight differentiable surrogate of action-conditioned force consequences and an input-dependent confidence estimate for residual-normalized guidance.

中文版本：

> 我们在 action expert 的 hidden states 上引入动作条件力后果预测头 $F_\phi$。给定当前观测、短历史力和候选动作 chunk，$F_\phi$ 预测未来力/力矩序列的因子化异方差高斯近似。该预测分布并不试图恢复接触过程的完整物理随机性，而是作为一个轻量、可微的动作条件力后果 surrogate，并为基于残差归一化的力引导提供输入相关的置信度估计。

更短的摘要表述：

> $F_\phi$ is a differentiable action-to-force consequence model. Its Gaussian output should be interpreted as a heteroscedastic predictive surrogate rather than a fully calibrated physical force distribution.

中文：

> $F_\phi$ 是一个可微的动作到力后果模型。它输出的高斯参数应被理解为异方差预测 surrogate，而不是完整校准的物理力分布。

## 10. 建议避免的表述

不建议写：

- “$F_\phi$ learns the true force distribution.”
- “Each time step has a ground-truth force distribution.”
- “The variance is the real physical variance.”
- “The model predicts the force target distribution from a single force label.”

建议写：

- “factorized heteroscedastic Gaussian predictive approximation”
- “action-conditioned force consequence model”
- “predictive uncertainty under partial observability”
- “input-dependent confidence for residual-normalized guidance”
- “differentiable surrogate for force-guided action sampling”

## 11. 总结

$F_\phi$ 确实为每个未来时间步输出一组力预测高斯参数。但这并不奇怪，前提是论文中把它解释为 **动作条件未来力的异方差预测近似**，而不是“单个标签本身的真实分布”。

这个设计与其他模块的关系是清晰的：

- local force conditioning 提供当前短时接触状态。
- action expert 生成动作条件 hidden states。
- $F_\phi$ 把这些 hidden states 解码成未来力后果的 predictive Gaussian。
- CFRG 使用 $F_\phi$ 的均值和方差进行残差归一化、目标校正和可微采样引导。

因此，在论文中站得住脚的核心表述应是：

$$
F_\phi
\text{ is an action-conditioned, heteroscedastic force consequence model for differentiable guidance.}
$$
