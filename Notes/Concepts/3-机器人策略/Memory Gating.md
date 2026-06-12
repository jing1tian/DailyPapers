---
type: concept
aliases: [记忆门控, 门控记忆机制]
---

# Memory Gating

## 定义
一种通过可学习的读门和写门选择性地读取和更新外部记忆库的机制，写门由技能边界检测触发，避免稠密历史聚合，实现紧凑且因果的任务状态保留。

## 数学形式

**读门**:
$$
c^m_t = \text{Attn}(W_q x_t, W_k M_t, W_v M_t), \quad \tilde{x}_t = x_t + \alpha^r_t W_m c^m_t
$$

**写门**:
$$
\alpha^w_t = \sigma(G_w(\tilde{x}_t, \hat{z}^h_t, \hat{b}_t)), \quad M_{t+1} = \begin{cases} U_\psi(M_t, \gamma_t) & \alpha^w_t > \eta \\ M_t & \text{otherwise} \end{cases}
$$

## 核心要点
1. 读门通过交叉注意力软加权检索历史记忆
2. 写门与技能边界对齐：边界处才写入，防止记忆无限增长
3. 两门均通过 L1 稀疏正则化保持稀疏激活
4. 实现全因果推理：仅依赖当前观测和已写入记忆

## 代表工作
- [[HiMem-WAM]]: 提出该机制，用于长视野机器人操作的历史状态保留

## 相关概念
- [[Hierarchical Chunking]]
- [[Hierarchical Latent Action]]
- [[World Action Model]]
- [[Attention Pooling]]
