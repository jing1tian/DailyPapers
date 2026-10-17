---
type: concept
aliases: [结构化转变槽位, Transition Slots, 物理槽位]
---

# Structured Transition Slots

## 定义

VLA-Δ 中将物理状态转变分解为 9 个预定义结构化类别的监督方案，每类槽位对应独立的预测头，提供从任务进度到接触状态的多粒度物理监督。

## 数学形式

$$
\mathcal{L}_{\text{slot}} = \sum_{i=1}^{9} w_i \mathcal{L}_i(\hat{y}_i, y_i)
$$

槽位预测头：

$$
u_{t,i} = W_i \cdot \text{LN}_i(\bar{z}_{t,g(i)}) + b_i
$$

## 核心要点

**9 个槽位类别**：
1. 任务进度（Task Progress）：多分类
2. 任务阶段（Task Phase）：多分类
3. 目标位置（Target Position）：回归
4. 夹爪位置（Gripper Position）：回归
5. 运动/相对位移（Motion/Displacement）：回归
6. 可交互物体位置（Interactable Position）：回归
7. 夹爪状态转变（Gripper State）：分类
8. 接触转变（Contact Transitions）：分类
9. 障碍物风险（Obstacle Risk）：分类

**结构化优势**：t-SNE 可视化显示不同 Query 组的隐状态形成清晰离散簇，微调后仍保持。

## 代表工作

- [[VLA-Delta]]: 提出结构化转变槽位，与光流和差异文本共同构成三路物理状态监督

## 相关概念

- [[Physical State Difference Queries]]
- [[Physical State Transition]]
- [[辅助监督]]
- [[多任务学习]]
