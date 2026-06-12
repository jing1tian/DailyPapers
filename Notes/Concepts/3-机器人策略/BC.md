---
type: concept
aliases: [Behavioral Cloning, 行为克隆, 模仿学习]
---

# BC（Behavioral Cloning）

## 定义
BC 是最简单的模仿学习方法，直接将专家演示数据作为监督信号，用最大似然估计训练策略网络拟合专家动作分布。

## 数学形式
$$\mathcal{L}_{BC} = -\mathbb{E}_{(s,a) \sim \mathcal{D}_{demo}} [\log \pi_\theta(a|s)]$$

## 核心要点
1. 本质是监督学习：输入状态，输出动作，拟合专家演示
2. 主要缺陷：分布偏移（covariate shift）——训练在专家轨迹上，推理时累积误差导致离轨
3. 现代 VLA 通常以 BC 为基础，再加 RL fine-tuning（如 RLHF、GRPO）提升泛化
4. Diffusion Policy、ACT、π0 等都是 BC 的现代变体

## 代表工作
- Ross et al. (2011)：DAgger，BC 的改进版本
- [[Diffusion Policy]]：以 BC 为目标的扩散策略
- [[ACT]]：以 BC 为目标的 transformer 策略
- [[SARM2]]：用 RL reward model 替代纯 BC 做自我改进

## 相关概念
- [[Diffusion Policy]]
- [[模仿学习]]
- [[IQL]]
