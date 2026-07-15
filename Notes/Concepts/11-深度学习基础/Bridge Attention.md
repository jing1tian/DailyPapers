---
type: concept
aliases: [Bridge Attention Mechanism, 桥接注意力]
---

# Bridge Attention

## 定义

Bridge Attention 是一种**多源注意力融合机制**，通过将自身 token、任务条件 token 和辅助查询 token 三路 key-value 沿 key 维度拼接后联合 softmax 归一化，实现多源信息流的可控融合；配合可学习门控 $\tanh(g)$ 对任务条件的注入强度进行自适应调节。

## 数学形式

**注意力权重**：

$$
\alpha = \text{softmax}\!\left(\frac{[\, Q K_{\text{self}}^\top,\; Q K_{\text{AQ}}^\top,\; \tanh(g)\, Q K_{\text{task}}^\top \,]}{\sqrt{d_h}}\right)
$$

**输出**：

$$
Z_{\text{bridge}} = \alpha \, [V_{\text{self}};\; V_{\text{AQ}};\; V_{\text{task}}]
$$

## 核心要点

1. **三路信息流**：Self（自注意力）、AQ（Action Query 动作引导）、Task（视觉-语言任务条件）
2. **门控稳定性**：$\tanh(g)$ 使训练初期任务条件弱影响模型，随训练收敛自动开放条件强度
3. **层次化注入**：与 VLM 各层隐藏状态层次对齐，低层注入几何信息，高层注入语义信息
4. **无需额外位置编码**：拼接方式天然保留各路 token 的独立表示

## 代表工作

- [[TS-MaskVLA]]: Bridge Attention 首次用于 VLA 动作专家的层次化条件注入

## 相关概念

- [[离散扩散]]
- [[时序-空间2D掩码]]
- [[VLA（视觉-语言-动作模型）]]
- [[Qwen]]
