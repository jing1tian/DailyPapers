---
type: concept
aliases: [指令锚定时序建模, IATM]
---

# Instruction-Anchored Temporal Modeling

## 定义

将每个时步的（视觉观测, 语言指令）对绑定为不可分割的**原子时序单元**，单元内使用双向注意力实现跨模态融合，单元间使用因果注意力聚合时序上下文，从而防止多帧拼接中的"指令遗忘"问题。

## 数学形式

原子时序单元：

$$
\mathbf{u}_{t}=(\mathbf{V}_{t},\, l_{t})
$$

单元内双向注意力（intra-pair）：

$$
\mathbf{h}_{\tau}=\mathrm{Attn}_{\mathrm{bi}}\left(\mathbf{V}_{\tau},\, l_{\tau}\right)
$$

单元间因果注意力（inter-pair）：

$$
\mathbf{o}_{t}=\mathrm{Attn}_{\mathrm{causal}}\left(\mathbf{h}_{t-T+1},\,\ldots,\,\mathbf{h}_{t}\right)
$$

整体通过[[Block Causal Attention|分块因果掩码]]在单个前向传播中实现。

## 核心要点

1. 将图文对视为原子单元，语言指令在每个时步都与对应视觉观测紧密绑定，不会被其他帧的视觉 token 稀释
2. 两级注意力层级：intra-pair（双向）保证跨模态融合，inter-pair（因果）保证时序一致性
3. 通过[[Block Causal Attention|分块因果注意力掩码]]无缝实现，无需修改模型结构或增加参数
4. 解决了窗口式 VLA 中"指令遗忘"（instruction forgetting）的核心问题

## 代表工作

- [[StreamPI]]: 提出该机制，在 [[π0.5]] 基础上实现零参数时序增强

## 相关概念

- [[Block Causal Attention]]
- [[KV Cache]]
- [[Random-Interval Streaming Training]]
- [[因果注意力]]
- [[双向注意力]]
