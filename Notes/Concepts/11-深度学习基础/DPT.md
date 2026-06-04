---
type: concept
aliases: [DPT, Dense Prediction Transformer Head]
---

# DPT（Dense Prediction Transformer）

参见：[[Dense Prediction Transformer]]

DPT 是用于密集预测任务（深度估计、语义分割）的 Transformer 架构，通过多尺度 Reassemble + Fusion 模块将 token 转换为像素级预测图。

在 GeoSem-WAM 中，DPT 风格辅助头被扩展为 3D 版本，用于对视频 DiT 的时空 token 施加几何和语义密集监督。
