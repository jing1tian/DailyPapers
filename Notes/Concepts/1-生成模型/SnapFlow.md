---
type: concept
aliases: [Snapshot Flow, ODE 步数压缩]
---

# SnapFlow

## 定义
对 flow matching 模型的推理加速方法，通过找到 ODE 轨迹上的"快照"点（关键中间状态），用少量 NFE（函数评估次数）重现完整轨迹，在不重训练的情况下压缩推理步数。

## 核心要点
1. 分析连续 ODE 轨迹，找到信息量最大的中间时间步
2. 用这些快照点构建多步跳跃的 ODE 求解方案
3. 推理时步数可从 100 步压缩到 4-8 步，质量损失有限
4. CrossVLA 在 VLA 连续动作生成中使用 SnapFlow 加速推理

## 代表工作
- [[SnapFlow]]：在 CrossVLA 中用于连续动作 flow matching 的推理加速

## 相关概念
- [[Flow Matching]]
- [[Conditional Flow Matching]]
- [[Diffusion Model]]
