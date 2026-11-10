---
type: concept
aliases: [IC-VFE, In-Context Value Function Estimator, 上下文价值函数估计器]
---

# IC-VFE（In-Context Value Function Estimator）

## 定义

以参考演示视频、语言指令和当前观测为输入，估计当前执行状态在任务中的相对进度（0–1）的轻量模型；无需测试时参数更新。

## 数学形式

$$
\hat{v}_t = 1 + \sum_{b=0}^{B-1} \nu_b \cdot p_{\phi,b}(o_t, \ell, \tau_{ctx})
$$

其中 $B = 201$ 为进度分档数，$\nu_b$ 为第 $b$ 档的分档值，$\tau_{ctx}$ 为参考演示上下文。

## 核心要点

1. **上下文条件化**：以参考演示的帧 token（patch-merging 压缩到 16 个 summary token）为条件，区别于不依赖参考的奖励模型
2. **架构**：Gemma 3 270M 解码器（18 层），在第 6、12、18 层使用 [[Block Causal Attention|分块因果注意力]]，实现跨块信息交换
3. **训练监督**：以 [[Robo-Dopamine]]（特权离线教师）的进度标签进行交叉熵训练

## 代表工作

- [[VICTR]]: IC-VFE 的提出论文，用于进度对齐的检索增强 VLA 上下文学习（OOD Kendall τ_b = 0.52，优于 RECAP 的 0.43）

## 相关概念

- [[Value Function]]
- [[RICL]]
- [[Block Causal Attention]]
- [[Robo-Dopamine]]
- [[In-Context Learning]]
