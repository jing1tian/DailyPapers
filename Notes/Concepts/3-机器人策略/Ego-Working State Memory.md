---
type: concept
aliases: [自我工作状态记忆, Ego-Working Memory, 意图记忆, 任务进度记忆]
---

# Ego-Working State Memory

## 定义

一种面向长时序机器人操作的任务进度追踪记忆机制，通过意图感知可学习查询聚合目标导向信息，并用冗余感知整合压缩时序相似的意图 token，使机器人能持续感知自身已完成的操作步骤，解决反应式 VLA 模型在多步任务中的"意图遗忘"问题。

## 数学形式

**意图感知查询（交叉注意力聚合）:**

$$
Z^{ego} = \operatorname{Softmax}\!\left(\frac{Q^{ego}K^T}{\sqrt{d}}\right)V
$$

**记忆库更新（冗余感知整合）:**

$$
\mathcal{M}^{ego}_t = \text{Cons}\!\left(\mathcal{M}^{ego}_{t-1} \cup \{Z^{ego}_t + \mathcal{E}_{temporal}(t)\}\right)
$$

**历史意图检索（交叉注意力）:**

$$
C^{ego}_t = \text{CrossAttn}(Z^{ego}_t, \mathcal{M}^{ego}_t, \mathcal{M}^{ego}_t)
$$

## 核心要点

1. **意图感知查询**: 4 个可学习 token $Q^{ego}$ 通过交叉注意力从视觉-语言 token 中聚合目标信息，编码当前时刻的操作意图
2. **时序位置编码**: 每个意图 token 加时序标记 $\mathcal{E}_{temporal}(t)$，使记忆库中不同时刻的意图可区分
3. **冗余感知整合 $\text{Cons}(\cdot)$**: 合并时序相邻且语义相似的意图 token，防止记忆库无限增长（默认记忆长度 16）
4. **动作生成条件**: 检索得到的 $C^{ego}_t$ 作为任务进度条件注入扩散变换器，同时也作为查询引导世界状态检索

## 代表工作

- [[AtlasVLA]]: 提出该机制，与持久世界状态记忆配合使用；消融实验显示移除此模块导致真实世界长时序任务下降 13.0%

## 相关概念

- [[Persistent World State Memory]]: 配套的空间感知记忆模块（解决感知遗忘，而此模块解决任务进度遗忘）
- [[VLA（视觉-语言-动作模型）]]: 上层任务框架
- [[Action Chunking]]: 动作生成策略，与自我工作记忆结合实现时序连贯动作
- [[Diffusion Transformer (DiT)]]: 接收自我工作上下文 $C^{ego}_t$ 作为条件的动作生成架构
