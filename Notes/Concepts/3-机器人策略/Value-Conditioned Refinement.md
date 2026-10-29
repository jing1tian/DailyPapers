---
type: concept
aliases: [VCR, 值条件精化]
---

# Value-Conditioned Refinement (VCR)

## 定义

一种策略微调方法，通过为每个轨迹帧分配 Hindsight 进度值标签，并将值 token 注入策略网络，使策略在推理时通过固定高值 token（v=1.0）向成功流形靠拢，无需在线故障检测。

## 数学形式

**失败帧可靠性衰减标签**:

$$
v_t = V(\tau_{fail}) \cdot \left(1 - \frac{t}{T}\right)^\alpha, \quad \alpha = 3.0
$$

**VCR 损失**:

$$
\mathcal{L}_{VCR} = -\mathbb{E}_{\tau \sim \mathcal{D}_{total}} \left[ \sum_t \log \pi_\theta(a_t \mid o_t, l, e_{val}(v_t)) \right]
$$

## 核心要点

1. **Hindsight 标签分配**: 成功/有效恢复帧 v=1.0，偏离恢复前缀 v=0.0，纯失败帧按 Eq.4 衰减。
2. **值 token**: MLP(v_t) 嵌入插入 [[π₀.₅]] 的输入流，无需架构大改。
3. **推理简洁**: 固定 v=1.0，无在线失败检测器或重试逻辑，零推理延迟。
4. 消融显示 α=3.0 最优；过快衰减（α=10）和过慢衰减（α=1）均降低性能。

## 代表工作

- [[RePO-VLA]]: 提出 VCR 作为两阶段训练的第二阶段

## 相关概念

- [[Recovery-Aware Initialization]]
- [[Progress-Aware Semantic Value Function]]
- [[Value Function]]
- [[π₀.₅]]
