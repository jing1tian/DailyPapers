---
type: concept
aliases: [Delta-Chunk, 增量动作块, Relative EEF Chunk]
---

# Delta-Chunk 动作表示

## 定义

以当前末端执行器（EEF）坐标系为参考，预测相对增量动作块 $^{G_t}T_{G_{t+k}}$，而非绝对世界坐标动作，从而将策略学习与机器人具体运动学解耦。

## 数学形式

固定底座臂平台映射：
$${}^W T_{G_{t+k}} = {}^W T_{G_t} \cdot {}^{G_t} T_{G_{t+k}}$$

浮动底座人形平台映射：
$${}^C T_{G_{t+k}} = ({}^W T_C)^{-1} \cdot {}^W T_{G_t} \cdot {}^{G_t} T_{G_{t+k}}$$

每臂动作维度：10 维（3D 平移 + 6D 旋转（SO(3) 矩阵前两行）+ 1D 夹爪）

## 核心要点

1. 增量动作相对于当前 EEF 帧，消除机器人绝对位姿的影响
2. 同一预训练 checkpoint 可通过简单平台映射 + IK 求解器部署到不同形态机器人
3. 固定底座臂与浮动底座人形使用不同但对称的映射公式

## 代表工作

- [[HyVLA-0.5]]: 提出并验证 Delta-Chunk 实现跨本体迁移（Track-B）

## 相关概念

- [[Action Chunking]]
- [[Mixture-of-Transformers]]
- [[Flow Matching]]
