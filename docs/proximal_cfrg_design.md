# CFRG: 基于同时间步观测力残差的闭环扩散引导

## 1. 为什么要再次重构

上一版 Proximal CFRG 的第 3 节使用了

$$
F^{ref}_{t+1}=F_t+(F_t-F_{t-1})
$$

作为近端力参考。这个设计虽然保持因果，但它的问题也很明显：力不是一个适合用一阶外推随便预测的量。接触力受几何、材料、控制器、速度、摩擦和瞬时接触状态共同影响，一阶外推容易让审稿人质疑：我们是不是在手工构造一个并不存在的未来力目标？

更合理的设定是利用闭环系统中真实存在的信号：

> 在时刻 $t$，$F_\phi$ 预测动作执行后的下一时刻力；到时刻 $t+1$，力传感器真实读数已经到达。此时预测力和真实力处在同一个物理时间步，可以直接计算残差，并用这个残差调节下一次动作扩散采样对力一致性的关注程度。

因此，新 CFRG 不再构造未来力参考，不再做一阶外推，也不声称知道未来力轨迹。它只做一件事：把已经发生的、同时间步的预测-观测误差变成下一轮 diffusion sampling 的闭环力反馈信号。

这个故事更容易站住脚，因为它同时满足三点：

- **因果性**：只使用已经观测到的传感器力，不访问未来。
- **时间对齐**：比较的是 $\mu_{t+1|t}$ 和 $F_{t+1}$，二者对应同一个物理时间步。
- **闭环性**：残差不被伪装成未来目标，而是作为反馈信号调节下一轮采样。

## 2. $F_\phi$ 的角色

$F_\phi$ 仍然是一个动作条件化的力后果模型：

$$
F_\phi(o_t,a_{t:t+H-1})
\rightarrow
\left(
\mu_{t+1:t+H|t},
\Sigma_{t+1:t+H|t}
\right)
$$

它回答的问题是：

$$
\text{如果在观测 }o_t\text{ 下执行候选动作 chunk，接下来每个时间步的力会如何？}
$$

训练时，$F_\phi$ 由真实未来力监督：

$$
\mathcal{L}_{force}
=
\sum_{h=1}^{H}
-\log
\mathcal{N}
\left(
F_{t+h};
\mu_{t+h|t},
\Sigma_{t+h|t}
\right)
$$

这不是说单个时间步的力本身是随机的多峰目标，而是说在部分观测、接触不确定和动作模型误差下，$F_\phi$ 输出的是一个异方差预测分布。均值表示预测的力后果，方差表示模型对该后果的置信度。这个表述与 noisy measurement / posterior-guided diffusion 中使用测量噪声尺度来加权一致性能量的思想一致 [2,3]。

## 3. 同时间步观测力残差

在控制周期 $t$，策略生成动作 chunk，并由 $F_\phi$ 给出第一个未来力分布：

$$
q_\phi(F_{t+1}|o_t,a_t)
=
\mathcal{N}
\left(
\mu_{t+1|t},
\Sigma_{t+1|t}
\right)
$$

机器人执行动作后，进入控制周期 $t+1$。此时传感器读到真实力：

$$
F^{obs}_{t+1}
$$

于是可以计算严格时间对齐的一步预测残差：

$$
e_{t+1}
=
F^{obs}_{t+1}
-
\mu_{t+1|t}
$$

以及标准化残差分数：

$$
r_{t+1}
=
e_{t+1}^{\top}
\Sigma_{t+1|t}^{-1}
e_{t+1}
$$

这里的关键点是：$F^{obs}_{t+1}$ 不是被估计出来的未来力，也不是由历史外推得到的参考。它就是下一控制周期已经真实接收到的传感器读数。换句话说，CFRG 使用的是 **observed next-force residual**，而不是 **predicted future force target**。

## 4. 残差如何用于引导

CFRG 不把 $e_{t+1}$ 当作一个要追踪的未来轨迹，也不把它加到某条手工构造的参考曲线上。残差只承担两个职责。

第一，它衡量当前策略-接触模型是否失配。如果 $r_{t+1}$ 很小，说明上一轮动作的力后果已经被 $F_\phi$ 解释，下一轮采样不需要额外强调力反馈。如果 $r_{t+1}$ 很大，说明当前接触状态出现了模型未解释的变化，下一轮动作生成应该更重视力一致性。

第二，它通过一个饱和门控产生采样时的 force guidance strength：

$$
\lambda_{t+1}
=
\lambda_{\max}
\frac{
r_{t+1}
}{
r_{t+1}+d_f
}
$$

其中 $d_f$ 是力/力矩维度。这个门控没有额外阈值、斜率、clip 或 residual gain；唯一需要调的量是 $\lambda_{\max}$，表示最大力反馈引导强度。

## 5. 观测力锚定的采样能量

在控制周期 $t+1$ 的 diffusion sampling 中，base policy 仍然提供动作先验。CFRG 只在采样过程中加入一个可微的力一致性能量。

在 denoising 第 $s$ 步，当前 noisy action 为 $x_s$，base model 输出速度场 $v_\theta$。先构造 clean action estimate：

$$
\hat{a}_0
=
x_s
-
s\cdot
\operatorname{stopgrad}
\left(
v_\theta(x_s,o_{t+1},s)
\right)
$$

然后用 $F_\phi$ 预测该候选动作的第一个力后果：

$$
\left(
\hat{\mu}_{t+2|t+1},
\hat{\Sigma}_{t+2|t+1}
\right)
=
F_\phi(o_{t+1},\hat{a}_0)_0
$$

由于推理时无法知道 $F_{t+2}$，CFRG 不构造 $F_{t+2}$ 的目标。它使用最新传感器读数 $F^{obs}_{t+1}$ 作为 **observed force anchor**，约束候选动作的近端力后果不要脱离当前真实接触状态：

$$
E_{CFRG}(x_s)
=
\lambda_{t+1}
\cdot
-\log
\mathcal{N}
\left(
F^{obs}_{t+1};
\hat{\mu}_{t+2|t+1},
\Sigma_{t+1|t}
\right)
$$

这一步不是在假设 $F_{t+2}=F_{t+1}$。更准确的说法是：在高频闭环执行中，最新力读数是当前接触状态的边界条件；当模型刚刚在这个边界条件上犯错时，下一轮采样应该优先选择其近端力后果与该边界条件相容的动作。下一次控制周期到来后，新的真实力会再次覆盖这个锚点，形成滚动闭环。

因此，CFRG 的本质不是 force tracking controller，而是 **residual-gated sensor anchoring**：

$$
\text{measured next-force residual}
\rightarrow
\text{guidance gate}
\rightarrow
\text{sensor-anchored proximal diffusion sampling}
$$

## 6. 为什么这个设计更合理

### 6.1 不再外推力

旧设计最容易被攻击的地方是 $F^{ref}_{t+1}=F_t+(F_t-F_{t-1})$。新设计完全删除这个假设。我们承认未来力未知，只使用已经发生的 $F^{obs}_{t+1}$ 来校验上一轮预测。

### 6.2 训练和推理时间对齐

训练时 $F_\phi$ 学的是：

$$
\mu_{t+1|t}
\leftrightarrow
F_{t+1}
$$

推理时 CFRG 计算的是：

$$
F^{obs}_{t+1}
-
\mu_{t+1|t}
$$

二者完全对齐。也就是说，训练监督中第一个未来力标签在推理闭环中变成了下一周期可观测的传感器反馈。这比“构造未来力参考”更严密。

### 6.3 与 receding-horizon diffusion policy 一致

Diffusion Policy 的核心优势之一是生成 action chunk，但以 receding-horizon 方式反复观测和重规划 [1]。CFRG 正是利用这个结构：上一轮 chunk 的第一个力预测在下一轮被真实传感器校验，然后校验误差影响下一轮采样。它不是在一个 open-loop chunk 内强行修正所有未来，而是在闭环边界上逐步修正。

### 6.4 与 measurement-guided diffusion 一致

Diffusion posterior sampling、Universal Guidance 和 FreeDoM 的共同思想是：base diffusion model 提供生成先验，外部可微能量在采样时提供条件约束或测量一致性 [2,3,4]。CFRG 的对应关系是：

- base action diffusion policy 是动作先验。
- $F_\phi$ 是可微 action-to-force measurement model。
- 传感器读数 $F^{obs}_{t+1}$ 是真实测量。
- 同时间步残差 $e_{t+1}$ 决定测量一致性能量是否需要被激活。

### 6.5 与接触丰富操作中的力反馈闭环一致

接触丰富任务不能只依赖视觉和 open-loop action chunk。Reactive Diffusion Policy 明确指出 chunk-level imitation policy 需要快速触觉/力反馈闭环来处理接触变化 [5]；TacDiffusion 和 Force Policy 也从不同角度强调了 force-domain action、力反馈和局部接触调节在精密操作中的价值 [6,8]；力反馈 MPC 工作则说明，在模型预测控制中直接纳入力传感器反馈可以弥补显式接触建模的不足 [7]。CFRG 的定位正是在 diffusion action policy 内部加入一个轻量的闭环力反馈通道。

## 7. 与三个模块的关系

整篇论文中，CFRG 只是第三个模块。三个模块可以这样串起来：

1. **Local force conditioning**  
   短历史力输入 action expert，让动作生成从一开始就感知当前接触状态。

2. **$F_\phi$ force consequence model**  
   给定候选动作，预测该动作将造成的未来力后果，并在训练时由真实未来力监督。

3. **CFRG closed-loop inference guidance**  
   推理时等待下一周期真实力到达，计算 $F_\phi$ 上一轮预测与传感器真实力之间的同时间步残差；当残差大时，在下一轮采样中增强力一致性能量，让动作头更关注力反馈。

这个链条可以写成：

$$
\text{force history}
\rightarrow
\text{contact-conditioned action prior}
\rightarrow
F_\phi \text{ predicts next force}
\rightarrow
\text{sensor verifies the same time step}
\rightarrow
\text{residual gates force-guided sampling}
$$

## 8. 推荐论文表述

英文推荐表述：

> CFRG uses a one-step delayed but time-aligned force prediction residual as a causal feedback signal. After executing the previous action chunk, the next force measurement becomes available and is compared with the force predicted for the same physical time step. The resulting normalized residual does not construct a future force trajectory; instead, it gates a sensor-anchored measurement-consistency energy during the next diffusion sampling process, making the action head attend more strongly to force consequences only when the predicted and observed contact responses disagree.

中文对应表述：

> CFRG 使用一个延迟一拍但时间严格对齐的力预测残差作为因果反馈信号。上一轮动作执行后，下一时刻真实力已经由传感器获得，并可与上一轮对同一物理时间步的预测力直接比较。该标准化残差不用于构造未来力轨迹，而是门控下一轮 diffusion sampling 中的观测力一致性能量，使动作头只在预测接触响应和真实接触响应不一致时更强地关注力后果。

## 9. 建议避免的表述

不建议写：

- “CFRG predicts the future force reference.”
- “CFRG extrapolates the next force from short history.”
- “The measured force is used as the ground-truth future force of the current sample.”
- “CFRG tracks a desired force trajectory.”

建议写：

- “time-aligned observed force residual”
- “one-step delayed causal force feedback”
- “residual-gated force-consistency guidance”
- “sensor-anchored proximal diffusion sampling”
- “$F_\phi$ serves as a differentiable action-to-force measurement model”

## 10. 最终方法摘要

上一轮预测：

$$
q_\phi(F_{t+1}|o_t,a_t)
=
\mathcal{N}
\left(
\mu_{t+1|t},
\Sigma_{t+1|t}
\right)
$$

下一轮观测到同一时间步真实力：

$$
e_{t+1}
=
F^{obs}_{t+1}
-
\mu_{t+1|t}
$$

标准化残差门控：

$$
r_{t+1}
=
e_{t+1}^{\top}
\Sigma_{t+1|t}^{-1}
e_{t+1},
\qquad
\lambda_{t+1}
=
\lambda_{\max}
\frac{r_{t+1}}{r_{t+1}+d_f}
$$

下一轮采样能量：

$$
E_{CFRG}
=
\lambda_{t+1}
\cdot
-\log
\mathcal{N}
\left(
F^{obs}_{t+1};
F_\phi(o_{t+1},\hat{a}_0)_0,
\Sigma_{t+1|t}
\right)
$$

一句话总结：

> CFRG 不预测未来参考力，而是把上一轮预测力与下一轮真实传感器力之间的同时间步误差，转化为下一轮扩散采样中的闭环力反馈强度。

## 11. 代码对应关系

当前代码应遵循以下实现：

- `policy.py` 存储上一轮 `predict_force(observation, actions)` 得到的 `prev_force_mu` 和 `prev_force_log_sigma`。
- 下一轮 `infer` 收到新观测力 `force` 后，取 `prev_force_mu[:, 0, :]` 与当前 `force` 计算同时间步残差。
- `force_prediction_error` 使用上一轮预测方差标准化。
- `force_target_mu` 直接使用当前真实传感器力 `force[:, None, :]` 作为 observed force anchor。
- 不再使用 `force + (force_t - force_{t-1})`，也不再构造任何一阶外推参考。
- `pi0.py` 中的 `sample_actions` 使用该 anchor 和门控强度，在 denoising 中通过 $F_\phi$ 的可微 NLL 对动作样本施加 force-consistency guidance。

## 12. 参考文献

[1] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin C. M. Burchfiel, Shuran Song. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**. Robotics: Science and Systems, 2023. [Project](https://diffusion-policy.cs.columbia.edu/), [RSS](https://roboticsconference.org/2023/program/papers/026/)

[2] Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye. **Diffusion Posterior Sampling for General Noisy Inverse Problems**. ICLR, 2023. [OpenReview](https://openreview.net/forum?id=OnD9zGAGT0k), [arXiv](https://arxiv.org/abs/2209.14687)

[3] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, Tom Goldstein. **Universal Guidance for Diffusion Models**. CVPR Workshops, 2023. [CVF](https://openaccess.thecvf.com/content/CVPR2023W/GCV/html/Bansal_Universal_Guidance_for_Diffusion_Models_CVPRW_2023_paper.html), [arXiv](https://arxiv.org/abs/2302.07121)

[4] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, Jian Zhang. **FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model**. ICCV, 2023. [arXiv](https://arxiv.org/abs/2303.09833)

[5] Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, Cewu Lu. **Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation**. Robotics: Science and Systems, 2025. [Project](https://reactive-diffusion-policy.github.io/), [arXiv](https://arxiv.org/abs/2503.02881)

[6] Yansong Wu, Zongxie Chen, Fan Wu, Lingyun Chen, Liding Zhang, Zhenshan Bing, Abdalla Swikir, Alois Knoll, Sami Haddadin. **TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation**. ICRA, 2025. [arXiv](https://arxiv.org/abs/2409.11047)

[7] Armand Jordana, Sebastien Kleff, Justin Carpentier, Nicolas Mansard, Ludovic Righetti. **Force Feedback Model-Predictive Control via Online Estimation**. ICRA, 2024. [HAL](https://hal.science/hal-04564888v1/document), [IEEE](https://ieeexplore.ieee.org/document/10611156/)

[8] Hongjie Fang, Shirun Tang, Mingyu Mei, Haoxiang Qin, Zihao He, Jingjing Chen, Ying Feng, Chenxi Wang, Wanxi Liu, Zaixing He, Cewu Lu, Shiquan Wang. **Force Policy: Learning Hybrid Force-Position Control Policy under Interaction Frame for Contact-Rich Manipulation**. RSS, 2026. [Project](https://force-policy.github.io/), [arXiv](https://arxiv.org/abs/2602.22088)
