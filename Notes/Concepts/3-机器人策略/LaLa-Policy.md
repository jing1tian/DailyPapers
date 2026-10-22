---
type: concept
aliases: [LaLa-Policy, LWM-initialized Diffusion Policy, 潜在世界模型初始化扩散策略]
---

# LaLa-Policy

## 定义

在 [[LaLa-World|LaLa-World]] 潜在世界模型预训练基础上微调得到的机器人操作策略，通过添加[[动作分块|动作查询（action queries）]]和扩散动作生成头，联合预测未来潜在状态和动作块，是世界模型预训练迁移到策略学习的代表方法。

## 数学形式

$$
(\hat{e}_{t+N},\ h_{t:t+N-1}) = P_\phi(E_\theta(x_t),\ \ell), \quad \hat{a}_{t:t+N-1} = A_\psi(\epsilon;\ h_{t:t+N-1})
$$

## 核心要点

1. **预训练迁移**: 直接从 LaLa-World 的 LWM 权重初始化，保留世界预测能力的同时学习动作生成
2. **联合预测**: 预测器（Qwen3.5-0.8B）同时输出未来潜在 $\hat{e}_{t+N}$ 和动作隐变量 $h$
3. **[[流匹配|Flow Matching]] 动作生成**: 动作头使用 DiT（16 层，宽度 1536）以[[流匹配|流匹配]]目标训练
4. **高效推理**: 不需要像素重建，推理速度比 [[Cosmos-Policy|Cosmos Policy]] 快 5.2×（LIBERO）和 5.5×（RoboCasa）

## 代表工作

- [[LaLa-World]]: LaLa-Policy 的来源论文

## 相关概念

- [[扩散策略|Diffusion Policy]]: 动作生成头的基础范式
- [[动作分块|Action Chunking]]: 动作以块（chunk）形式预测
- [[JEPA]]: LaLa-World 基于的预训练范式
- [[Cosmos-Policy|Cosmos Policy]]: 主要对比基线
