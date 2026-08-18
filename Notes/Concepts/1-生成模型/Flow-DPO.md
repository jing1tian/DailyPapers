---
type: concept
aliases: [Reference-Relative Flow-DPO, Continuation DPO, Flow Matching DPO]
---

# Flow-DPO

## 定义

Flow-DPO（参考相对 Flow-DPO）是将 [[DPO|直接偏好优化]] 适配到 [[Flow Matching]] 框架的偏好学习方法：以候选动作的 Flow 期望误差为能量，以冻结参考策略为基准计算偏好变化，避免策略过度偏离原始分布。

## 数学形式

候选 $A$ 的 Flow 能量（含两阶段 scaffold/residual）：

$$
\mathcal{E}_\theta(A|\chi) = \frac{1}{2}\bigl[\mathcal{E}_\theta^C(C|\chi) + \mathcal{E}_\theta^D(D|C,\chi)\bigr]
$$

参考相对偏好差：

$$
\Delta_\theta = \bigl[\mathcal{E}_\theta(A^-) - \mathcal{E}_\theta(A^+)\bigr] - \bigl[\mathcal{E}_\text{ref}(A^-) - \mathcal{E}_\text{ref}(A^+)\bigr]
$$

偏好损失（$q$ 为置信权重，$\beta$ 为温度）：

$$
\mathcal{L}_\text{pref} = -q \log \sigma(\beta \Delta_\theta)
$$

完整训练目标：

$$
\mathcal{L} = \mathcal{L}_\text{pref} + \lambda_+ \mathcal{E}_\theta(A^+) + \lambda_\text{rep} \mathcal{L}_\text{replay}
$$

## 核心要点

1. **参考相对**：以冻结参考策略的能量差为基准，防止策略偏离太远（类比标准 DPO 的 KL 惩罚项）。
2. **可迁移性**：偏好 objective 可直接植入 π₀.₅、Legato、RTC 等宿主的原生 Flow 能量，无需修改架构，只替换目标函数。
3. **同纤维排名**：候选对在共享行为条件 $\chi_{r,k}$ 下采样，确保偏好排名不改变任务语义。
4. **能量-概率对应**：Flow 能量越低 = 宿主策略兼容性越高（近似等价于对数似然）；偏好候选以 $\mathcal{E}_\theta(A^+)$ 项额外锚定支持。
5. **SFT vs DPO**：直接 SFT 到偏好候选会抑制运动或切换任务相位；DPO 通过相对排名在保持 SR 的同时显著降低 handoff 代价。

## 代表工作

- [[BICPO-VLA]]：提出参考相对 Flow-DPO，用于 VLA 异步接管偏好优化；验证迁移到 π₀.₅、Legato、RTC

## 相关概念

- [[DPO]]
- [[Flow Matching]]
- [[行为条件化动作纤维]]
- [[Haar动作分解]]
