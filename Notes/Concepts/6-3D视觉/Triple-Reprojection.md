---
type: concept
aliases: [三重重投影, Triple Reprojection Constraint]
---

# Triple-Reprojection

## 定义
Triple-Reprojection 是一种用于大基线 novel view synthesis 的几何约束机制，通过在源视图、参考视图和目标视图之间建立三重一致性约束，确保生成的新视角在几何上自洽。

## 核心要点
1. 扩展了经典的双目重投影（epipolar constraint），加入第三个视角的约束
2. 统一处理旋转视差和平移视差，适合大基线场景
3. 与 video diffusion model 结合，用几何约束引导生成过程
4. 减少大视角变化下的 ghosting artifact 和几何不一致

## 代表工作
- [[UniWorld-View]]: 在大基线 NVS 中使用 Triple-Reprojection 约束

## 相关概念
- [[NeRF]]
- [[3DGS]]
- [[VGGT]]
- [[SEVA]]
