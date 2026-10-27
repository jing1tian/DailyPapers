---
type: concept
aliases: [动作分块, Action Chunk, Temporal Action Abstraction]
---

# Action Chunking

## 定义

动作分块（Action Chunking）是机器人模仿学习中的一种时间抽象表示：策略一次预测并执行连续 $H$ 步（一个"chunk"）的动作序列，而非逐步预测单步动作，从而减少对精确时间步对齐的依赖并平滑短时震荡。

## 数学形式

$$
\pi_\theta(\mathbf{o}_t) \mapsto \hat{\mathbf{a}}_{t:t+H-1} \in \mathbb{R}^{H \times d_a}
$$

其中 $H$ 为动作块长度，$d_a$ 为单步动作维度。

## 核心要点

1. **去抖动**: 一次预测多步消除了逐步策略中常见的高频抖动（compounding jitter）。
2. **时间对齐鲁棒性**: 减少演示数据中帧级对齐误差对训练的影响。
3. **与扩散/Flow 兼容**: 扩散策略（Diffusion Policy）、Flow Matching Policy 等通常在动作块上建模分布。
4. **Chunk 长度权衡**: $H$ 过小失去平滑效果，$H$ 过大导致对环境变化响应迟钝。

## 代表工作

- ACT（Action Chunking with Transformers）: 首次系统性引入 action chunking 于模仿学习
- [[MAST-VLA]]: 使用 $H=8$ 步动作块 + 64 个动作查询 token

## 相关概念

- [[Delta-Chunk 动作表示]]
- [[Hierarchical Chunking]]
- [[VLA（视觉-语言-动作模型）]]
