---
type: concept
aliases: [Wan Move, 轨迹条件化视频生成]
---

# Wan-Move

## 定义
Wan-Move 是基于 Wan 2.2 视频生成模型的轨迹条件化扩展版本，通过稀疏轨迹点（如末端执行器路径）引导视频生成，常用作机器人世界模型的基线。

## 核心要点
1. 以稀疏轨迹坐标（而非像素掩码）作为控制信号条件化视频生成
2. 在分布内场景（已见本体、标准夹爪）表现与像素掩码方法相近
3. 遭遇未见本体或自定义夹爪时泛化性弱于像素对齐条件化方法

## 与 Masked Visual Actions 的对比
- **Wan-Move**：条件在稀疏轨迹点上，与本体结构相关，跨本体迁移能力有限
- **Masked Visual Actions**：条件在像素空间遮罩上，本体无关，跨本体迁移更鲁棒

## 代表工作
- [[MaskedVisualActions]]: 在 DROID、BEHAVIOR、Real-World 三个测试集上将 Wan-Move 作为基线进行对比

## 相关概念
- [[Wan 2.2 A14B]]
- [[视觉掩码动作]]
- [[前向动力学模型]]
