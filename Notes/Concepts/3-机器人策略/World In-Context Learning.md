---
type: concept
aliases: [WICL, 世界上下文学习]
---

# World In-Context Learning (WICL)

## 定义

一种学习范式：冻结权重的基础模型从演示出发，在世界（如仿真器）中排练，将反馈写入 context 而非权重，获得可跨场景迁移的语义知识。

## 数学形式

$$
C_0 = I_\theta(d),\quad (\tau_k, r_k) \sim W^{\pi_\theta(\cdot|o,C_k)},\quad C_{k+1} = U_\theta(C_k, \tau_k, r_k)
$$

部署时以冻结 $C^*$ 对未见实例执行：$a_t \sim \pi_\theta(\cdot | o_t, C^*)$

## 核心要点

1. **知识存储在 context**：与权重学习不同，所学技能不绑定训练分布
2. **世界提供反馈**：弥补纯 ICL 缺乏世界反馈的缺陷
3. **语义可迁移**：context 只记录对象角色、任务步骤等语义，不记录视觉外观
4. **支持递归自我改进**：context 可读写，agent 可在冻结权重下持续改进

## 代表工作

- [[RoboScript]]: WICL 在机器人操作中的实现，50 任务 83.4% 成功率

## 相关概念

- [[In-Context Learning]]
- [[Task Memory]]
- [[Recursive Self-Improvement]]
- [[Reinforcement Learning]]
