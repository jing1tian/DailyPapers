---
type: concept
aliases: [Coefficients-Preserving Sampling, 系数保留采样]
---

# CPS（Coefficients-Preserving Sampling）

## 定义
一种 DDIM 风格的扩散模型随机采样策略，在注入探索噪声时精确保持边际分布与确定性调度路径一致，避免传统 SDE 转化中过量噪声导致的 artifacts 和梯度不稳定问题。

## 数学形式

给定 rectified flow 的速度预测 $\hat{v}_\theta$，CPS 在步骤 $t_i \to t_{i+1}$ 的随机转移为：

$$
\hat{x}_0 = x_{t_i} - t_i\hat{v}_\theta, \quad \hat{\epsilon} = x_{t_i} + (1-t_i)\hat{v}_\theta
$$

$$
x_{t_{i+1}} = \mu_\theta + s_i\epsilon, \quad \mu_\theta = (1-t_{i+1})\hat{x}_0 + \sqrt{t_{i+1}^2 - s_i^2}\,\hat{\epsilon}
$$

其中 $s_i = t_{i+1}\sin(\eta\pi/2)$，$\epsilon \sim \mathcal{N}(0, I)$，$\eta \in [0,1]$ 控制探索强度。

**关键性质**：$(t_{i+1}^2 - s_i^2) + s_i^2 = t_{i+1}^2$，总噪声量精确匹配调度系数。

对数似然（简化形式）：

$$
\log \pi_\theta(x_{t_{i+1}} | x_{t_i}) \propto -\|x_{t_{i+1}} - \mu_\theta\|^2
$$

## 核心要点
1. **保持边际分布**：注入噪声后样本仍在确定性采样路径的边际分布上，奖励始终在干净视频上评估
2. **避免 SDE 问题**：SDE 转化在高噪声时步的扩散系数爆炸，需要额外 clipping；CPS 从结构上规避这一问题
3. **探索强度可控**：$\eta=0$ 退化为确定性采样，$\eta=1$ 为最大噪声注入

## 代表工作
- Flash-GRPO (He et al., arXiv 2605.15980)
- [[LingBot-Video]]: 在单步 GRPO 随机探索中使用 CPS

## 相关概念
- [[Flash-GRPO]]
- [[GRPO]]
- [[Rectified Flow]]
- [[DDIM]]
