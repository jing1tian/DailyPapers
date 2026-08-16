---
type: concept
aliases: [World Token, Training-Time World Modeling]
---

# World Tokens

## 定义
一种"训练时世界建模、部署时零开销"的 embodied policy 设计：在训练阶段通过 World Adapter 插入世界模型预测，推理时完全移除 world model 组件，只保留其对策略表征的塑造效果。

## 核心要点
1. **World Adapter**：轻量适配器，在训练时引入未来状态预测 token
2. **Training-Time Only**：world model 只在训练参与，推理时 drop 掉，部署速度与纯 VLA 相同
3. 在 LIBERO、SIMPLER 和 R1 Pro 真实机械臂上验证
4. 与 [[JEPA-WAM]] 的对比：JEPA-WAM 训练推理都有 world model，World Tokens 推理时无

## 核心权衡
- 优点：部署开销为零，接近 VLA 的推理速度
- 代价：可能无法充分利用推理时的 world model 规划能力（与 [[TempoWAM]] 等 test-time WAM 相比）

## 代表工作
- World Tokens: Enhancing Embodied Policies with Training-Time World Modeling (2026)

## 相关概念
- [[JEPA-WAM]]
- [[WorldVLA]]
- [[WAM]]
