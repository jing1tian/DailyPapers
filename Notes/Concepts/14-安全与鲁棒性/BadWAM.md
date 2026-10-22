---
type: concept
aliases: [BadWAM, World-Action Drift Attack]
---

# BadWAM

## 定义

BadWAM 是首个专门针对 [[WAM|World Action Model]] 的测试时查询攻击框架，揭示了 WAM 特有的"世界动作漂移"漏洞：小幅视觉扰动可在预测未来保持合理的同时，单独劫持动作输出导致任务失败。

## 数学形式

$$
\max_{\|\delta\|_\infty \leq \varepsilon} D_{\text{act}}(a^\delta, a) \quad \text{s.t.} \quad D_{\text{img}}(z^\delta, z) \leq \tau_{\text{img}}
$$

## 核心要点

1. **查询型黑盒攻击**：无需目标模型梯度，通过零阶优化（17次查询/步）在线生成扰动
2. **双变体**：纯动作攻击（最强效果）和保形攻击（最强隐蔽性）
3. **世界动作漂移**：动作偏移与失败高度相关，而想象漂移不是失败的可靠信号
4. **区别于 TAPDreamer**：需要实时查询目标模型；TAPDreamer 完全离线，无需任何目标策略访问

## 代表工作

- [[BadWAM]]（论文笔记）: 原始论文，Qi Li et al., NUS, 2026

## 相关概念

- [[WAM]]
- [[World-Action Drift]]
- [[TAPDreamer]]
- [[对抗补丁攻击]]
