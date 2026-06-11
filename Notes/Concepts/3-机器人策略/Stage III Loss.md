---
type: concept
aliases: [记忆策略微调损失, HiMem-WAM Stage III Loss]
---

# Stage III Loss

## 定义
HiMem-WAM Stage III 中记忆条件化策略的综合训练目标，将动作预测、高层规划、低层执行、边界检测和记忆门控五个损失加权求和。

## 数学形式

$$
\mathcal{L}_{\text{ft}} = \mathcal{L}_{\text{act}} + \alpha_h \mathcal{L}_{\text{plan}} + \alpha_l \mathcal{L}_{\text{exec}} + \alpha_b \mathcal{L}_{\text{bd}} + \alpha_m \mathcal{L}_{\text{gate}}
$$

## 核心要点
1. 动作损失 L_act 是最终目标，其余为辅助任务
2. 规划和执行损失保持层次结构的一致性
3. 边界损失和门控损失确保记忆机制的正确学习
4. 五项联合训练是三阶段流水线的最终端到端优化

## 代表工作
- [[HiMem-WAM]]: Stage III 训练目标

## 相关概念
- [[Memory Gate Loss]]
- [[Memory Gating]]
- [[Stage II Loss]]
