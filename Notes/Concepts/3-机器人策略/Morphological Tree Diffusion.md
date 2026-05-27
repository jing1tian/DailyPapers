---
type: concept
aliases: [MPTD, 形态树扩散, 跨形态树搜索]
---

# Morphological Tree Diffusion

## 定义

将蒙特卡洛树搜索（MCTS）与扩散模型结合，利用异构机器人形态在相同任务上的动作数据作为条件信号，通过树状搜索结构发掘不同末端执行器间的行为相关性，从而在去噪过程中实现跨形态知识迁移。

## 数学形式

跨形态节点价值评估：
$$v=\begin{cases}-\|\mathbf{x}-\mathbf{y}\|^{2}+M_{0},&\text{if }\text{dim}(\mathbf{x})=\text{dim}(\mathbf{y})\\ -\|\mathbf{x}_{1:d_{\min}}-\mathbf{y}_{1:d_{\min}}\|^{2}+M_{1},&\text{if }\text{dim}(\mathbf{x})\neq\text{dim}(\mathbf{y})\end{cases}$$

## 核心要点

1. **四阶段迭代**：Selection（选择自去噪或跨形态条件）→ Expansion（EBF 条件下生成候选动作段）→ Simulation（快速跳跃去噪评估）→ Backpropagation（回传价值）
2. **节点提取**：从在相同任务上执行的异构形态中，通过最近邻匹配提取元动作节点
3. **维度差异处理**：不同维度形态仅比较公共维度（$d_{\min}$），加基础奖励偏置 $M_1$ 补偿
4. **消融贡献**：去掉 MPTD 后性能下降约 10.3%（67.2% → 56.9%）

## 代表工作

- [[X-DiffVLA]]: 提出 MPTD，将 Monte Carlo Tree Diffusion 扩展到跨形态场景

## 相关概念

- [[蒙特卡洛树搜索]]
- [[扩散模型]]
- [[Embodied Forcing]]
- [[视觉语言动作模型]]
