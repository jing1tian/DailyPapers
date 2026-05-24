---
type: concept
aliases: [自回归策略, Autoregressive Action Generation]
---

# Autoregressive Policy（自回归策略）

## 定义

以自回归方式逐 token 预测动作序列的机器人策略架构，将连续动作离散化后以语言模型风格生成。

## 核心要点

1. 将连续动作空间离散化（如 VQ-VAE 或直接分箱）
2. 与 [[Flow Matching]] 和 [[Diffusion Policy]] 的连续生成方式形成对比
3. 代表模型：[[Pi0-FAST]]、[[OpenVLA]]、[[RT-2]]

## 代表工作

- [[COAST]]: 在自回归 VLA [[Pi0-FAST]] 上验证了 Conceptor 引导的跨架构泛化性

## 相关概念

- [[VLA]]
- [[Flow Matching]]
- [[Diffusion Policy]]
- [[Pi0-FAST]]
