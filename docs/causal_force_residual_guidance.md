# Causal Force Residual Guidance: 用预测力残差驱动的因果力引导

## 1. 背景问题

在力引导策略中，一个自然目标是利用真实力反馈修正动作生成过程。原始思路可以概括为：模型预测未来力，机器人在线测得真实力，然后根据两者差距进行力引导。

但这里有一个关键矛盾：

- 训练时可以看到未来标签 $F_{t+1:t+H}$，因此可以监督力预测器 $F_\phi$。
- 推理时只能观测到当前及过去的力 $F_{\le t}$，不能知道未来真实力 $F_{t+1:t+H}$。
- 如果推理时直接令未来目标力等于当前力 $F_t$，则隐含假设为 $F_{t+1:t+H} \approx F_t$。这个假设只适合恒力保持任务，对泵送、插入、挤压、接触建立等动态接触任务并不充分。

因此，力引导模块不应该依赖未来真实力，也不应该简单把当前力当作整段未来力目标。更合理的做法是：把当前真实力作为对上一轮未来预测的闭环校正信号。

## 2. 核心想法

本文档提出 **Causal Force Residual Guidance, CFRG**。

CFRG 的核心是把“预测力和真实力之间的差距”定义为一个因果力残差：

$$
\epsilon_t = F_t - \mu^{t-1}_{0}
$$

其中：

- $F_t$ 是当前时刻真实测得的力。
- $\mu^{t-1}_{0}$ 是上一轮在时刻 $t-1$ 预测的第一个未来力均值。它理论上对应当前时刻 $t$ 的力。
- $\epsilon_t$ 不是未来误差，而是一个已经发生、可在线观测的预测残差。

直觉上，$\epsilon_t$ 描述了模型对真实接触动力学的预测失配。如果上一轮预测的力与当前真实力差距很小，说明当前动作计划和接触状态仍然可信；如果差距很大，说明接触发生了变化，或者模型对当前环境动力学估计不准，此时应增强力引导。

## 3. 与已有方法的关系

CFRG 可以理解为以下几类思想的结合：

- **Receding-horizon diffusion policy**：Diffusion Policy 使用动作 chunk 并以 receding horizon 方式在线执行，因此天然存在“上一轮预测、本轮观测、本轮重规划”的时间结构。
- **Energy-guided diffusion sampling**：FreeDoM 和后续 gradient guidance 工作表明，可以在 diffusion 采样过程中加入外部 energy 的梯度，从而在不重新训练主模型的情况下引导生成结果。
- **Forward prediction loss guidance**：Gradient guidance 的优化视角强调，用前向预测损失构造 guidance 比直接硬约束样本更自然，因为它保留了预训练生成分布的结构。
- **Tactile/force servoing**：机器人力控和触觉伺服中常用实时接触误差进行闭环修正。CFRG 将这一思想移植到动作 diffusion 的采样过程中。

相关参考：

- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023: https://arxiv.org/abs/2303.04137
- FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.pdf
- Gradient Guidance for Diffusion Models: An Optimization Perspective: https://arxiv.org/html/2404.14743v2
- Reactive Diffusion Policy: https://reactive-diffusion-policy.github.io/
- TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation: https://arxiv.org/html/2409.11047v1

## 4. 训练阶段

训练阶段仍然使用未来力标签监督力预测器。给定当前观测 $o_t$ 和专家动作 chunk $a_{t:t+H-1}$，力预测头输出未来力分布：

$$
F_\phi(o_t, a_{t:t+H-1}) =
\left(
\mu_{t+1:t+H},
\log \sigma_{t+1:t+H}
\right)
$$

其中每个 horizon step 都对应一个 6D force/torque 分布：

$$
p_\phi(F_{t+h} \mid o_t, a_{t:t+H-1})
=
\mathcal{N}
\left(
F_{t+h};
\mu_h,
\operatorname{diag}(\sigma_h^2)
\right)
$$

训练损失为对角高斯负对数似然：

$$
\mathcal{L}_{force}
=
\frac{1}{H}
\sum_{h=1}^{H}
\frac{1}{2}
\left[
(F_{t+h} - \mu_h)^\top
\Sigma_h^{-1}
(F_{t+h} - \mu_h)
+ \log |\Sigma_h|
+ d \log 2\pi
\right]
$$

这里未来力只用于监督 $F_\phi$，不作为推理输入，因此不造成信息泄漏。

## 5. 推理阶段：Force Residual

在时刻 $t-1$，策略生成动作 chunk，并用力预测器得到上一轮未来力分布：

$$
\left(
\mu^{t-1}_{0:H-1},
\Sigma^{t-1}_{0:H-1}
\right)
$$

其中 $\mu^{t-1}_{0}$ 预测的是下一控制周期的力，也就是当前时刻 $t$ 的力。

到了时刻 $t$，机器人真实观测到 $F_t$。于是可以计算标准化预测误差：

$$
r_t
=
(F_t - \mu^{t-1}_{0})^\top
\left(\Sigma^{t-1}_{0}\right)^{-1}
(F_t - \mu^{t-1}_{0})
$$

$r_t$ 是一个 Mahalanobis residual score。它表示当前真实接触状态相对于上一轮模型预测有多“意外”。

## 6. 用 Residual 调节引导强度

CFRG 不应该在任何时候都强力引导。若模型预测和真实力一致，则主策略已经工作良好；若预测误差变大，才需要增强力引导。因此引导强度定义为：

$$
\lambda_t
=
\lambda_{max}
\cdot
\operatorname{sigmoid}
\left(
k (r_t - \tau_0)
\right)
$$

其中：

- $\lambda_{max}$ 控制最大引导强度。
- $k$ 控制门控曲线斜率。
- $\tau_0$ 是触发引导的误差阈值。

这个设计的含义是：

- $r_t \ll \tau_0$：预测可信，引导弱。
- $r_t \approx \tau_0$：接触状态开始偏离，引导逐渐增强。
- $r_t \gg \tau_0$：接触动力学明显失配，引导强。

## 7. 为什么需要因果未来力参考

需要先区分两个概念：

- $F_\phi$ 是 **动作到力后果模型**。给定观测和候选动作，它预测“如果执行这个动作，未来力可能是什么”。
- 力引导还需要一个 **参考力轨迹**。只有存在参考，才能定义候选动作的力后果是否偏离期望。

因此，$F_\phi$ 本身不能直接给出 guidance 目标。如果把同一个候选动作的 $F_\phi$ 输出当作自己的目标，损失会变成自洽项，无法告诉采样器应该往哪个方向修正动作。

推理时又不能访问真实未来力 $F_{t+1:t+H}$。CFRG 采用一个因果、闭环的参考构造：把上一轮预测轨迹向前平移，得到当前时刻可用的未来力先验：

$$
\bar{\mu}^{t}_{0:H-1}
=
\operatorname{shift}
\left(
\mu^{t-1}_{1:H-1}
\right)
$$

实际实现中，最后一个 step 可以复制上一轮最后一个预测：

$$
\bar{\mu}^{t}_{h}
=
\begin{cases}
\mu^{t-1}_{h+1}, & h < H - 1 \\
\mu^{t-1}_{H-1}, & h = H - 1
\end{cases}
$$

然后用当前 residual 对整段未来力参考做闭环校正：

$$
\tilde{\mu}^{t}_{h}
=
\bar{\mu}^{t}_{h}
+ K \cdot \operatorname{clip}
\left(
F_t - \mu^{t-1}_{0}
\right)
$$

其中 $K$ 是 residual gain。为了避免力传感器尖峰或模型方差过小导致过强修正，clip 在标准差单位下执行：

$$
\operatorname{clip}(\epsilon_t)
=
\sigma^{t-1}_{0}
\cdot
\operatorname{clip}
\left(
\frac{\epsilon_t}{\sigma^{t-1}_{0}},
-c,
c
\right)
$$

最终得到的 $\tilde{\mu}^{t}_{0:H-1}$ 是一个因果未来力参考。它不是未来真实力，也不是 $F_\phi$ 重新预测出的标签，而是“上一轮预测 + 当前真实反馈”的闭环参考轨迹。它的作用是在采样时提供一个可比较的 reference，使当前候选动作的 $F_\phi$ 预测结果可以被拉向更符合当前真实接触反馈的区域。

如果任务本身能提供外部期望力，例如恒力控制目标、显式阻抗控制器目标，或单独训练的 desired-force head，则可以直接把该外部目标作为 `force_target_mu/log_sigma`，不需要使用上述 residual-corrected reference。CFRG 的构造主要用于没有显式未来力目标、但有在线真实力反馈的场景。

## 8. Diffusion 采样中的力引导 Energy

在 diffusion 采样第 $s$ 步，当前 noisy action 为 $x_s$。模型预测 flow velocity $v_\theta(x_s, o_t, s)$，从而得到 clean action estimate：

$$
\hat{a}_0
=
x_s - s \cdot \operatorname{stopgrad}
\left(
v_\theta(x_s, o_t, s)
\right)
$$

再用力预测器估计该动作会产生的未来力分布：

$$
\left(
\hat{\mu}_{0:H-1},
\hat{\Sigma}_{0:H-1}
\right)
=
F_\phi(o_t, \hat{a}_0)
$$

定义力引导 energy：

$$
E_{CFRG}(x_s)
=
\lambda_t
\sum_{h=0}^{H-1}
\frac{1}{2}
\left[
(\tilde{\mu}^{t}_{h} - \hat{\mu}_{h})^\top
\tilde{\Sigma}_{h}^{-1}
(\tilde{\mu}^{t}_{h} - \hat{\mu}_{h})
+ \log |\tilde{\Sigma}_{h}|
\right]
$$

然后对 noisy action 求梯度：

$$
g_s
=
\nabla_{x_s} E_{CFRG}(x_s)
$$

在 flow matching 采样中，模型从噪声积分到动作。当前实现中采样步长为负，因此将 gradient 加到 velocity 上：

$$
v_s^{guided}
=
v_s + g_s
$$

这会使 clean action estimate 沿着降低 force prediction energy 的方向移动。

## 9. 为什么这个设计更合理

### 9.1 因果性

CFRG 只使用：

- 当前观测 $o_t$
- 当前真实力 $F_t$
- 历史力 $F_{\le t}$
- 上一轮模型预测的未来力分布

它不使用未来真实力 $F_{t+1:t+H}$，因此不存在信息泄漏。

### 9.2 解决当前力等于未来目标的问题

旧设计若默认使用 $F_t$ 作为未来目标，相当于假设：

$$
F_{t+1:t+H} = F_t
$$

CFRG 则使用上一轮预测轨迹构造未来力参考，并由当前真实反馈修正：

$$
F_{t+1:t+H}^{target}
\approx
\operatorname{shift}
\left(
\mu^{t-1}_{1:H}
\right)
+ K \epsilon_t
$$

这个假设更适合动态接触任务，因为它允许未来力继续沿着任务进程演化，而不是被固定为当前力。

### 9.3 统一预测误差、方差和引导

预测误差不是简单欧氏距离，而是用预测方差归一化：

$$
r_t
=
\epsilon_t^\top
\Sigma^{-1}
\epsilon_t
$$

因此：

- 如果模型本来就不确定，误差不会过度放大。
- 如果模型很确定但预测错了，引导会明显增强。
- 方差具有明确语义：它是力预测器的不确定性，而不是短历史力窗口的统计波动。

### 9.4 与机器人闭环控制一致

CFRG 的结构接近 Kalman filter / model predictive control / tactile servoing：

$$
\text{prediction}
\rightarrow
\text{measurement}
\rightarrow
\text{residual update}
\rightarrow
\text{corrected plan}
$$

这比单纯把力作为额外 observation 更主动，因为力反馈不仅影响下一次网络前向，还直接改变 diffusion sampling 的优化方向。

## 10. 当前代码映射

当前实现中：

- `Policy` 保存上一轮完整的 `force_mu` 和 `force_log_sigma`。
- 下一轮推理时，用当前真实力和上一轮第 0 步预测计算 `force_prediction_error`。
- 同时构造 `force_target_mu` 和 `force_target_log_sigma`，传入 `sample_actions`。
- `Pi0.sample_actions` 接收 `force_prediction_error`，并用它计算 $\lambda_t$。
- `Pi0.sample_actions` 支持 `[B, F]` 和 `[B, H, F]` 两种 force target，因此可以做整段 horizon 的力轨迹引导。
- `cst_tau` 保留为旧参数名兼容，但语义上应逐步替换为 `force_prediction_error`。

## 11. 可写进论文的方法名

推荐名称：

**Causal Force Residual Guidance (CFRG)**

可选副标题：

**Causal Residual-Corrected Force Guidance for Contact-Rich Diffusion Policies**

一句话描述：

CFRG uses the one-step force prediction residual as a causal feedback signal to adaptively guide action diffusion toward residual-corrected future force distributions.

中文描述：

CFRG 将上一轮力预测与当前真实力之间的差值视为因果力残差，用该残差自适应调节 diffusion 采样中的力引导强度，并校正未来力参考分布，从而实现无需未来真实力的闭环力引导。

## 12. 建议实验

### 12.1 消融实验

建议至少比较：

- No force guidance：只使用基础 pi0。
- Current-force guidance：把当前力 $F_t$ 当作未来目标。
- Error-gated current-force guidance：用预测误差调节强度，但目标仍是当前力。
- CFRG：使用 shift prediction + residual correction。

### 12.2 指标

建议报告：

- 任务成功率。
- 峰值力 $F_{max}$。
- 力方差 $\operatorname{Var}(F)$。
- force prediction NLL。
- prediction residual score $r_t$ 的时间曲线。
- 接触建立阶段、稳定接触阶段、异常接触阶段的分段指标。

### 12.3 关键可视化

建议画三条曲线：

- 真实力 $F_t$
- 上一轮预测力 $\mu^{t-1}_{0}$
- CFRG 校正后的未来力参考 $\tilde{\mu}^{t}_{0:H-1}$

如果 CFRG 有效，应能看到：

- 接触突变时 $r_t$ 上升。
- $\lambda_t$ 随 $r_t$ 自适应增强。
- 校正后的未来力参考比单纯当前力目标更平滑、更符合任务阶段。

## 13. 局限性

CFRG 仍依赖力预测器 $F_\phi$ 的质量。如果 $F_\phi$ 学得不好，residual 和 future force reference 都会不可靠。

此外，CFRG 默认当前 residual 可以对未来 horizon 产生近似平移式修正：

$$
\tilde{\mu}^{t}_{h}
=
\bar{\mu}^{t}_{h}
+ K \epsilon_t
$$

这是一阶近似。对于高度非线性的接触动力学，可以进一步扩展为 learned correction：

$$
\Delta \mu_{0:H-1}
=
G_\psi
\left(
o_t,
F_{t-L:t},
\epsilon_t,
\mu^{t-1}_{0:H-1}
\right)
$$

也就是说，可以训练一个小的 residual adapter，让模型学习不同任务阶段下 prediction residual 应该如何传播到未来力参考。

## 14. 总结

CFRG 的核心贡献是把力引导从“使用当前力作为未来目标”改为“使用当前真实力校正上一轮未来力预测，并形成因果参考”。这样既保持因果性，又保留未来力轨迹的动态结构。

简洁公式为：

$$
\epsilon_t = F_t - \mu^{t-1}_{0}
$$

$$
r_t = \epsilon_t^\top \left(\Sigma^{t-1}_{0}\right)^{-1} \epsilon_t
$$

$$
\lambda_t = \lambda_{max} \operatorname{sigmoid}(k(r_t - \tau_0))
$$

$$
\tilde{\mu}^{t}_{0:H-1}
=
\operatorname{shift}
\left(
\mu^{t-1}_{1:H}
\right)
+ K \cdot \operatorname{clip}(\epsilon_t)
$$

$$
E_{CFRG}
=
\lambda_t
\sum_{h=0}^{H-1}
\operatorname{NLL}
\left(
\tilde{\mu}^{t}_{h};
F_\phi(o_t, \hat{a}_0)_h
\right)
$$

这使力引导成为一个闭环、因果、可解释的 contact residual correction 模块。
