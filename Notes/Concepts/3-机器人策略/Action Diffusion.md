---
type: concept
aliases: [动作扩散, Diffusion Policy]
---

# Action Diffusion（动作扩散）

## 定义

Action Diffusion 是将扩散模型（Diffusion Model）或 Flow Matching 应用于机器人动作生成的方法，将动作序列的生成建模为去噪过程，以当前观测为条件从噪声中生成动作块。

## 数学形式

Flow Matching 形式下的动作生成：

$$
a^{\sigma} = (1-\sigma)a^{\text{gt}} + \sigma \epsilon_a, \quad \mathcal{L}_{\text{act}} = \|\hat{v}_a - v_a\|_2^2
$$

其中 $v_a = \epsilon_a - a^{\text{gt}}$ 为 flow target。

## 核心要点

1. 相比确定性策略，扩散/流匹配能更好地建模多模态动作分布
2. 结合 [[Action Chunking]]，每次生成 $H$ 步动作序列以提升时间一致性
3. Diffusion Policy（Chi et al., 2023）是早期代表工作
4. WAM 系列（Fast-WAM、GeoSem-WAM）使用 Flow Matching 而非 DDPM 进行动作生成

## 代表工作

- [[GeoSem-WAM]]: 使用 Flow Matching 的 Action DiT 生成动作块

## 相关概念

- [[Flow Matching]]
- [[Action Chunking]]
- [[World Action Model]]
