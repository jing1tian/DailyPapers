---
type: concept
aliases: ["LeapBot World-Action"]
---

# LeapBot-WA

## 定义
基于 JEPA 表示的机器人 world-action 策略，在 JEPA 特征空间里做未来预测和动作生成。

## 数学形式
$$\hat{s}_{t+1} = f_\phi(s_t, a_t), \quad a_t = \pi_\theta(s_t)$$

## 核心要点
1. JEPA 特征的预测性表示比视频 latent 更适合控制
2. 是 La-JEPA 的直接对比方法

## 代表工作
- 见关联笔记

## 相关概念
- [[JEPA-WAM]]
- [[La-JEPA]]
- [[V-JEPA 2.1]]
