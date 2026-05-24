---
type: concept
aliases: [General Navigation Model, GNM]
---

# GNM（General Navigation Model）

## 定义
Shah et al. (2022) 提出的通用视觉导航模型，在大规模跨机器人、跨场景导航数据上预训练，学习从当前观测和目标图像预测导航动作的通用策略。

## 核心要点
1. 以图像对（当前帧、目标帧）为输入，输出导航动作（线速度、角速度）
2. 跨机器人平台泛化：在多种移动机器人数据上联合训练
3. 不使用显式地图，纯视觉端到端策略
4. 在 RECON benchmark 上 ATE=1.87，是传统导航基线

## 代表工作
- [[CoME]]: RECON benchmark 上的对比基线之一

## 相关概念
- [[世界模型]]
- [[RECON]]
