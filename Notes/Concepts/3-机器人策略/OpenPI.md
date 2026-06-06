---
type: concept
aliases: [OpenPI, π₀, pi0, Physical Intelligence]
---

# OpenPI

## 定义
Physical Intelligence（π₀）发布的开源大型机器人基础模型，基于 flow matching 的 VLA，在大规模多样化机器人数据上预训练，是目前最广泛使用的 flow-matching 机器人策略基线之一。

## 数学形式
$$\pi_\theta: (\mathbf{o}_t, \ell) \rightarrow \mathbf{a}_{t:t+H}$$
Flow matching 目标：$\mathcal{L} = \mathbb{E}_{t,\mathbf{a}} \| v_\theta(\mathbf{a}_t, t|\mathbf{o}, \ell) - (\mathbf{a} - \mathbf{a}_t) \|^2$

## 核心要点
1. 基于 [[流匹配]]（Flow Matching）生成动作 chunk，而非离散化 token 输出
2. 在 Bridge v2、OXE、内部数据等多源数据上预训练，具备强泛化能力
3. 是 [[VLA]] 领域的重要开源 baseline，众多后续工作（SimpleVLA、FlowPRO 等）以其为对比基准
4. 与 [[OpenVLA]] 对比：OpenPI 基于连续 flow matching，OpenVLA 基于离散 token 预测

## 代表工作
- Black et al. (2024): π₀ 原始论文 (Physical Intelligence)
- [[FlowPRO]]: 基于 OpenPI 架构做 RL 微调
- [[SimpleVLA]]: 以 OpenPI 作为多步扩散基线对比

## 相关概念
- [[流匹配]]
- [[VLA]]
- [[OpenVLA]]
