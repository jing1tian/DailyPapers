---
type: concept
aliases: [KeyWorld Robot World Model]
---

# KeyWorld

## 定义

KeyWorld 是一种机器人具身世界模型，通过姿态驱动的 RDP（Ramer-Douglas-Peucker）算法提取关键帧，再用视频生成模型合成稀疏帧序列，用于策略训练数据增强。

## 核心要点

1. **姿态驱动选帧**: 基于末端执行器轨迹的几何简化（RDP 算法）确定关键时刻
2. **局限性**: 未显式建模夹爪开合等接触事件，导致关键操作时刻可能漏检
3. **对比 SKIP**: SKIP 在 KeyWorld 基础上加入多模态事件感知和动作条件插帧，显著提升 GripCov

## 代表工作

- [[SKIP]]: 直接与 KeyWorld 对比，证明事件感知选帧的必要性

## 相关概念

- [[世界模型 (World Model)]]
- [[核时间分割 (KTS)]]
