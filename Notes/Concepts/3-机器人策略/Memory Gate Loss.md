---
type: concept
aliases: [记忆门控损失, Gate Loss]
---

# Memory Gate Loss

## 定义
HiMem-WAM 中用于训练记忆读写门控的损失函数，通过将写门与技能边界对齐并对读写门施加稀疏正则化，引导门控机制在关键时刻选择性地更新记忆。

## 数学形式

$$
\mathcal{L}_{\text{gate}} = \text{BCE}(\alpha^w_t, \bar{b}_t) + \lambda_r \|\alpha^r_t\|_1 + \lambda_w \|\alpha^w_t\|_1
$$

## 核心要点
1. BCE 项将写门与技能边界标签对齐，确保边界处写入
2. 读门 L1 正则化鼓励稀疏记忆检索，专注相关历史
3. 写门 L1 正则化防止过于频繁的记忆更新
4. 三项联合约束使记忆机制既有效又高效

## 代表工作
- [[HiMem-WAM]]: Stage III 训练目标的一部分

## 相关概念
- [[Memory Gating]]
- [[Hierarchical Chunking]]
