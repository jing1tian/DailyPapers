---
type: concept
aliases: [Wan2.1, Wan video model]
---

# Wan

## 定义
阿里巴巴开源的大规模视频生成基础模型系列（如 Wan2.1），基于 [[Diffusion Transformer|DiT]] 架构，提供强大的视频动力学先验，被广泛用作 [[World Action Model|World Action Model]] 的预训练骨干网络。

## 核心要点
1. 提供高质量的视频生成先验，WAM 类工作（如 [[Tactile-WAM]]）在其基础上改造为联合预测视觉/动作（及触觉）的去噪骨干
2. 常见改造方式：将骨干变为"因果"（causal）结构以支持自回归式 rollout，并扩展 token 序列以容纳动作/状态/触觉等模态
3. 在多个 WAM 工作中作为"视觉动力学先验来源"反复出现（如 [[Causal-rCM]] 中的 Wan2.1-1.3B/14B 蒸馏实验）

## 代表工作
- [[Tactile-WAM]]: 使用 "Wan/WAM 兼容去噪骨干"（Causal Wan DiT）联合去噪视觉、触觉、动作 token
- [[Causal-rCM]]: 对 Wan2.1 进行流式蒸馏的相关工作

## 相关概念
- [[Diffusion Transformer]]
- [[World Action Model]]
- [[Video DiT|Video Diffusion Transformer]]
