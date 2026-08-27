---
type: concept
aliases: [CoTracker, Co-Tracker]
---

# CoTracker

## 定义
CoTracker（Karaev et al., NeurIPS 2023）：通过 transformer 同时追踪视频中多个点的像素轨迹，利用点间相关性提升追踪精度和鲁棒性。

## 核心要点
1. Joint tracking：同时追踪所有 query 点，利用点间 co-visibility 约束
2. Sliding window 机制处理长视频，每次处理固定长度窗口
3. 输出密集点轨迹，可作为 visual track 信号提供给 robot policy

## 代表工作
- [[CoTracker3]]：第三代版本，精度进一步提升
- [[TrAct]]：用 CoTracker 生成 visual tracks 作为 VLA 的中间表示

## 相关概念
- [[光流 (Optical Flow)]]
- [[点轨迹追踪]]
- [[TrAct]]
