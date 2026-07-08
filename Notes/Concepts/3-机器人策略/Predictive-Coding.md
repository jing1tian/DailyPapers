---
type: concept
aliases: [预测编码, Predictive Coding in Robotics]
---

# Predictive Coding（预测编码）

## 定义
神经科学理论：大脑通过预测感觉输入并只处理"预测误差"（残差）来高效编码信息，而非处理原始信号。在机器人学中被用来设计稀疏模态融合机制。

## 数学形式
$$\delta = x_{actual} - x_{predicted}$$

其中 $\delta$ 为预测误差（prediction error），只有当 $|\delta|$ 超过阈值时才激活下游处理。

## 核心要点
1. 只关注"意外"刺激，抑制可预测输入——降低信息冗余
2. 在机器人触觉融合中：先预测期望触觉，只将残差（实际 - 预期）注入 VLA
3. 避免了 modality collapse（高带宽视觉特征淹没稀疏触觉信号）

## 代表工作
- [[ResTacVLA]]：将 Predictive Coding 用于触觉-视觉融合，提出 SAG（Surprise-Aware Gating）

## 相关概念
- [[SAG]]
- [[Tactile World Model]]
- [[Contact-Rich Manipulation]]
