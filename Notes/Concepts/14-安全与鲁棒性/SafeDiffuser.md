---
type: concept
aliases: [Safe Diffusion, 安全扩散模型]
---

# SafeDiffuser

## 定义
将控制屏障函数（CBF）与扩散模型生成过程结合，使扩散模型在生成轨迹时满足安全约束，适用于安全关键的机器人控制和 RL 场景。

## 数学形式
在扩散去噪过程中引入 CBF 梯度投影：
$$\hat{x}_{t-1} = x_{t-1} - \eta \nabla_{x} h(x)|_{x=x_{t-1}}$$
其中 $h(x) \geq 0$ 是安全集指示函数。

## 核心要点
1. **推理时约束**：不修改训练，在推理时投影梯度保证安全
2. **CBF 集成**：利用 [[CBF]] 的不变集理论提供理论保证
3. **扩展到多智能体**：被 CBF-Diffusion 等工作扩展到 MARL 场景

## 代表工作
- SafeDiffuser 原始论文: 单智能体安全扩散
- CBF-Guided Diffusion (2606.12640): 扩展到多智能体

## 相关概念
- [[CBF]]
- [[MADIFF]]
- [[扩散模型]]
