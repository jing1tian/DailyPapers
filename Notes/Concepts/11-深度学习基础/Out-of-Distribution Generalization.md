---
type: concept
aliases: [OOD泛化, OOD Generalization, Out-of-Distribution, 分布外泛化]
---

# Out-of-Distribution Generalization

## 定义

Out-of-Distribution（OOD）泛化指模型在训练数据分布以外的输入上保持良好性能的能力，是衡量模型鲁棒性和泛化能力的重要维度。

## 核心要点

1. **定义对立面**: In-Distribution（ID）评测使用与训练集同分布的样本；OOD 评测使用分布外样本
2. **OOD 类型**:
   - **属性 OOD**: 改变颜色、形状、材质等视觉属性（模型通常较鲁棒）
   - **位置 OOD**: 改变物体摆放位置
   - **语义 OOD**: 改变任务指令语义（如动作计数值）
   - **组合 OOD**: 同时改变多个维度
3. 预训练 VLA 模型相比从头训练的[[模仿学习]]模型有更强的 OOD 泛化能力

## 代表工作

- [[VLA-REPLICA]]: 设计颜色/形状/计数三类 OOD 评测，揭示 VLA 在计数任务上的 OOD 泛化失败

## 相关概念

- [[Sim-to-Real Gap]]
- [[Domain Randomization]]
- [[VLA]]
