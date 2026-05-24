---
type: concept
aliases: [Imagination with Auto-Regression over an Inner Speech, token-based world model]
---

# IRIS

## 定义
基于 token 的 Transformer 世界模型，将图像 tokenize 为离散 token，用自回归 Transformer 预测未来 token，再解码为图像，结合 RL 做视觉环境的 model-based 规划。

## 数学形式
$$s_{t+1} \sim p_\theta(s_{t+1} | s_t, a_t), \quad s = \text{VQ-tokenize}(o)$$

## 核心要点
1. 用 VQ-VAE 将观察图像压缩为离散 token 序列
2. 自回归 Transformer 在 token 空间预测下一帧
3. 解码器重建图像用于 RL agent 的虚拟环境训练
4. 在 Atari 等视觉 RL benchmark 上有强竞争力

## 代表工作
- [[IRIS]]：Micheli et al. 2023，token-based world model 原作
- [[ITC]]：Identifiable Token Correspondence，改进 IRIS 的 token 时序一致性问题

## 相关概念
- [[DreamerV3]]
- [[STORM]]
- [[World Model]]
- [[VQ-VAE]]
