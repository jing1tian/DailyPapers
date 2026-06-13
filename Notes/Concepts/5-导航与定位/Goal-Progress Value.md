---
type: concept
aliases: [目标进度值, 目标进展值, goal progress, value function for navigation]
---

# Goal-Progress Value

## 定义

衡量智能体距离导航目标接近程度的标量信号，归一化到 $[0, 1]$（0=最远，1=到达），作为辅助监督信号帮助策略学习长程目标感知能力。

## 数学形式

$$
v_{t+H} = \mathrm{clip}\!\left(1 - \frac{\|p_{\mathrm{end}} - p_t\|_2}{d_{\max}},\ 0,\ 1\right)
$$

- 真实数据：使用欧氏距离 $\|p_{\mathrm{end}} - p_t\|_2$
- 仿真（HM3D）：使用测地线距离（考虑障碍物的最短路径长度）
- $d_{\max}$：训练轨迹长度的上百分位数，用于归一化

## 核心要点

1. **长程规划辅助**: 提供目标方向的全局信息，弥补局部观测（[[partial observability|部分可观测性]]）的不足
2. **消融验证**: 在 [[NavWAM]] 中，加入价值监督使 h=8 的 ATE 从 0.287 降至 0.192（-33%），对长时域预测关键
3. **未来扩展**: 可用于推理时的 Best-of-N 价值引导采样——通过采样多个动作候选并选择价值最高者
4. **距离类型自适应**: 室外/简单场景用欧氏距离，仿真复杂场景用测地线距离

## 代表工作
- [[NavWAM]]: Goal-Progress Value 的系统性应用，作为 Canvas 第 8 帧的预测目标

## 相关概念
- [[Latent Canvas]]: Goal-Progress Value 在 Canvas 中占据专用预测帧（帧 8）
- [[Navigation World Model]]: Goal-Progress Value 可与世界模型结合用于规划评分
- [[partial observability]]: Goal-Progress Value 解决的核心问题之一
