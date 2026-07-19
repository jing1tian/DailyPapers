---
type: concept
aliases: [保形攻击, 想象保形攻击, Stealth WAM Attack]
---

# Imagination-Preserving Attack（保形攻击）

## 定义
保形攻击是 [[BadWAM]] 提出的针对 [[WAM]] 的对抗攻击变体：在最大化动作空间偏移的同时，通过 Lagrangian 惩罚项约束预测未来的漂移量，使攻击在视觉层面保持隐蔽性。

## 数学形式

$$
\delta_t^* = \arg\max_{\|\delta_t\|_\infty \leq \varepsilon} \Bigl[ D_{\text{act}}(a_{t:t+H-1}^\delta,\ a_{t:t+H-1}) - \lambda\, D_{\text{img}}(z_{t+1:t+K}^\delta,\ z_{t+1:t+K}) \Bigr]
$$

## 核心要点
1. **强度-隐蔽性权衡**：$\lambda$ 控制动作破坏力与想象隐蔽性的 tradeoff，$\lambda = 0$ 退化为纯动作攻击
2. **Lagrangian 松弛**：将想象漂移约束松弛为软惩罚，便于无约束优化
3. **实证效果**：在 [[LIBERO]] 上，保形攻击以接近纯动作攻击的强度（相差约 2 pp）实现显著更低的想象漂移
4. **检测绕过**：基于增强一致性的检测器在保形攻击下，5% 误报率时召回率仅 13–21%，无效

## 代表工作
- [[BadWAM]]: 提出保形攻击作为 [[World-Action Drift]] 框架的隐蔽性变体

## 相关概念
- [[World-Action Drift]]: 保形攻击所利用的底层漏洞
- [[WAM]]: 攻击目标模型类别
- [[Zeroth-Order Gradient Estimation]]: 实现保形攻击的黑盒优化方法
- [[对抗补丁攻击]]: 传统对抗攻击（无隐蔽性设计）
