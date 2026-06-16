---
type: concept
aliases: [RPRO, Proximalized Preference Optimization, PRO, FlowPRO]
---

# RPRO 损失（Proximalized Preference Optimization）

## 定义

一种无奖励模型的离线偏好优化损失，在流匹配策略框架下通过对比胜出/失败动作对进行策略优化，并附加近端正则项防止奖励幅度爆炸（reward hacking）。

## 数学形式

隐式奖励代理：
$$r_\theta(s,a) = \frac{\beta}{2}\left(\ell_{ref}(s,a) - \ell_\theta(s,a)\right)$$

偏好优化损失（PRO）：
$$\mathcal{L}_{PRO}(\theta) = -\mathbb{E}\left[\log\sigma(r_\theta(s,a^w) - r_\theta(s,a^l)) + \frac{1}{2}\sum_{a\in\{a^w,a^l\}}\left[\log\sigma(r_\theta(s,a)) + \log\sigma(-r_\theta(s,a))\right]\right]$$

组合损失：
$$\mathcal{L}_{RPRO}(\theta) = \lambda_{PRO}\mathcal{L}_{PRO}(\theta) + \lambda_{SFT}\mathcal{L}_{SFT}(\theta)$$

## 核心要点

1. 无需单独的奖励模型或价值函数，直接用流匹配损失差作为隐式奖励
2. 第一项为对比项：拉近策略预测与胜出动作，推离失败动作
3. 第二项为近端正则项：对称惩罚奖励绝对幅度，防止偏离参考策略过远
4. 与 SFT 损失联合优化，混合比例随迭代轮次调整（80/20 → 70/15/15）

## 代表工作

- [[HyVLA-0.5]]: 提出 FlowPRO，在四项精细操作任务上达到 94–99% 成功率

## 相关概念

- [[DPO]]
- [[Flow Matching]]
- [[SERL]]
