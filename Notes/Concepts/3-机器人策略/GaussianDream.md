---
type: concept
aliases: [GaussianDream, Gaussian Dream]
---

# GaussianDream

## 定义
一种基于 [[3DGS|3D Gaussian Splatting]] 的前馈世界模型 VLA 方法，通过 [[VGGT]]/[[TGE]] 构建 1024-token 前缀，训练时提供当前帧重建与未来帧预测的密集几何监督，推理时保留前缀条件化动作生成。

## 核心要点
1. 用 [[VGGT]] 重建当前帧 3D 几何，用 [[TGE]] 建模短视野 Gaussian 演化
2. 从演示轨迹自动挖掘 RGB、深度、伪 3D 场景流等多模态监督信号
3. 非对称训练-推理设计：训练时全量重建+预测，推理时丢弃解码头，保留 GaussianDream 前缀
4. 前缀规模为 1024 个 token，运行时需要 VGGT/TGE 几何路径，推理延迟约 531ms

## 核心局限
- 1024-token 前缀携带纠缠的状态、动态和动作信息
- 需要专用运行时几何路径（VGGT/TGE），部署开销高
- 被 [[GaussianDream++]] 以 20 个策略原生 Token 替代，延迟降至 330ms

## 代表工作
- [[GaussianDream]]（论文）: GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation (2026)
- [[GaussianDream++]]: 后续工作，将 1024 token 前缀压缩为 20 个策略原生 World Token

## 相关概念
- [[3DGS]]
- [[VGGT]]
- [[TGE]]
- [[GaussianDream++]]
- [[VLA（视觉-语言-动作模型）]]
