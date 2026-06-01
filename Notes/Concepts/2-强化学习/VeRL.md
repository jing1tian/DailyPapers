---
type: concept
aliases: [Versatile RL, VeRL-Tool]
---

# VeRL

## 定义
字节跳动开源的 LLM 强化学习训练框架，支持大规模分布式 PPO/GRPO 训练，设计上将 actor rollout 与 critic/reward 计算解耦，支持工具调用 (tool-use) 场景的 agent RL。

## 数学形式

GRPO 目标（VeRL 中常用）：
$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} \hat{A}_i - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

## 核心要点
1. 支持 FSDP + tensor parallel 混合并行，可扩展到 70B+ 模型
2. VeRL-Tool 扩展支持 agent 工具调用场景的 rollout 生成
3. 与 vLLM 推理引擎无缝集成，提升 rollout 吞吐
4. ProRL Agent 中被列为对比基线框架

## 代表工作
- [[ProRL]]: 对比 VeRL-Tool 与 ProRL Agent 架构

## 相关概念
- [[SkyRL]]
- [[NeMo]]
- [[GRPO]]
