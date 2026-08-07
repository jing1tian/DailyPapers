---
type: concept
aliases: [Geometry-Aware Latent Distillation, GLaD VLA]
---

# GLaD

## 定义

GLaD（**G**eometry-Aware **La**tent **D**istillation）是一种 7B 参数的 VLA 方法，通过在骨干多个中间层将 VLA 特征与场景级几何特征（VGGT）进行蒸馏对齐，提升机器人操作的 3D 感知能力。

## 核心要点

1. **多层几何蒸馏**: 不仅在输出层，而是在骨干 transformer 的多个中间层植入几何监督信号
2. **指令无关**: GLaD 的对齐目标是全场景几何特征，不区分语言指令所指的目标物体（指令无关）
3. **性能**: LIBERO 平均成功率 94.1%（7B 参数骨干），与 π₀（94.2%）相当
4. **Mind-VLA 对比**: Mind-VLA 用 345M 参数（1/20× 大小）达到 93.9%，同时通过指令感知对齐解决 GLaD 的指令无关局限

## 代表工作

- [[Mind-VLA]]: 将 GLaD 的场景级对齐提升为指令感知目标物体对齐

## 相关概念

- [[VGGT]]
- [[Vision-Language-Action]]
- [[Spatial Forcing]]
- [[辅助监督]]
