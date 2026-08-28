---
type: concept
aliases: [World Representation Head, 世界表示解码头, WRH]
---

# World Representation Head

## 定义
在 VLA 训练阶段附加的轻量解码头，将 [[World State Tokens]] 和 [[World Prediction Tokens]] 的隐状态解码为显式 3D Gaussian 基元，提供当前世界重建和未来世界预测的稠密几何监督；推理时完全移除，零部署开销。

## 数学形式

$$
(G_t,\, \{\widehat{G}_{t+h}\}_{h \in \mathcal{H}}) = W_\psi(H_t^S, H_t^P)
$$

## 核心要点
1. **训练专用**：仅在训练阶段存在，推理时与渲染器、辅助目标一起完全丢弃
2. 包含特征场展开解码器 $D_\text{state}$、[[几何-外观分解|几何-外观分支]] $H_\text{geo}/H_\text{app}$、位移预测网络 $H_\Delta$
3. 通过可微分 [[3DGS|Gaussian Splatting]] 渲染 RGB、度量深度和覆盖率，提供多模态密集监督
4. 监督信号约束 World Token 学习物理结构，防止表示坍塌为纯动作特征

## 代表工作
- [[GaussianDream++]]: 首次提出此设计，实现训练-推理路径完全解耦

## 相关概念
- [[World State Tokens]]
- [[World Prediction Tokens]]
- [[几何-外观分解]]
- [[耦合未来预测]]
- [[3DGS]]
- [[GaussianDream++]]
