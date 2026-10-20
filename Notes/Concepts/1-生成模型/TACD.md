---
type: concept
aliases: [Teacher-Anchored Consistency Distillation, 教师锚定一致性蒸馏]
---

# TACD（Teacher-Anchored Consistency Distillation）

## 定义

TACD 是在[[Consistency Distillation|标准一致性蒸馏（CD）]]基础上增加"教师端点锚定损失"$\mathcal{L}_{TA}$ 的训练方法，通过监督学生速度场与多步教师展开端点的均匀速度对齐，解决 CD 的局部-全局误差脱钩问题，实现近无损单步生成。由 [[RealtimeWAM]] 提出。

## 数学形式

**教师端点均匀速度**（K 步教师展开）：

$$
u_T(a_t, t) = \frac{a_t - a_0^T}{t}
$$

**教师锚定损失**：

$$
\mathcal{L}_{TA} = \mathbb{E}\left[ \left\| v_S(a_t, t) - \text{sg}[u_T(a_t, t)] \right\|^2 \right]
$$

**总损失**：

$$
\mathcal{L} = \mathcal{L}_{CD} + \lambda \cdot \mathcal{L}_{TA}, \quad \lambda = 0.2
$$

**误差界**：

$$
\mathbb{E}\|e_{\text{total}}\|^2 \leq 2 \cdot \mathcal{L}_{TA} + 2 \cdot \mathbb{E}\|\delta\|^2
$$

其中 $\delta$ 为教师数值积分误差，以 $O(1/K^2)$ 随 K 收缩。

## 核心要点

1. **全局误差修正**：CD 仅最小化局部一致性误差，TACD 额外约束全局端点对齐，显著减小一步推理的端点偏差。
2. **K=10 最优**：教师展开步数 K=10 在性能和训练开销间取得最优平衡，K=20 提升极小。
3. **冻结视频专家**：三个动作专家（学生、教师、EMA）共享同一冻结视频 KV 缓存，峰值显存从 65.98 GiB 降至 24.73 GiB。
4. **训练开销**：增加约 50% 训练时间（5h 21min → 8h 3min on 16×H100），推理时无额外开销。
5. **等价加权端点损失**：$\mathcal{L}_{TA}$ 等价于权重为 $1/t^2$ 的加权端点损失，对低噪声区域给予更强约束。

## 代表工作

- [[RealtimeWAM]]（Lv et al., 2026）：提出 TACD，首次在 MoT-WAM 上实现近无损单步动作生成

## 相关概念

- [[Consistency Distillation]]
- [[Flow Matching]]
- [[MeanFlow]]
- [[RealtimeWAM]]
- [[CEWP]]
