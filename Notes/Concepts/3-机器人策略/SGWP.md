---
type: concept
aliases: [Self-Guided World Policy space, 几何感知策略空间]
---

# SGWP (Self-Guided World modeling in geometry-aware Policy space)

## 定义
SG-WAM 提出的 latent 空间设计：同时满足几何感知（利用 3D 几何结构）和动作对齐（对动作生成有直接意义）的 WAM 预测空间，以 JEPA 风格在其中预测未来状态。

## 核心要点
1. 现有 WAM 的两难：像素空间预测（计算重）vs 任意 latent 预测（与动作生成不对齐）
2. SGWP 用 [[VGGT]] 引入几何锚点（metric depth、点云结构），使 latent 具有物理意义
3. [[JEPA]] 风格预测：在特征空间预测而非像素，避免 decoder 计算开销
4. "Self-Guided"：世界模型预测的 latent 被显式约束为对策略输出有用（joint training）

## 代表工作
- [[SG-WAM]]: 提出 SGWP，NUS Marcelo Ang 组，LIBERO + 真机验证

## 相关概念
- [[JEPA]]
- [[VGGT]]
- [[WAM-Survey]]
