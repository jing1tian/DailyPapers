---
type: concept
aliases: [链式注意力, 结构化注意力掩码, Chain Attention Mask]
---

# Chain Attention

## 定义
链式注意力是一种**结构化单向注意力掩码**机制，通过预先定义不同信息流之间的有向连接，控制多流 Transformer 中各模态之间的信息传播方向，防止信息短路。

## 数学形式

在 PrismWAM 中，第 $j$ 层的注意力路由规则：

$$
\text{RGB} \to \text{Depth} \to \text{Action}
$$

具体掩码：
- $h_R^{(j+1)} = B_V^{(j)}(h_R^{(j)} \mid C, \sigma_R)$ （RGB 只读自身 + 上下文）
- $h_D^{(j+1)} = B_V^{(j)}(h_D^{(j)} \mid h_{R,\leq 0}^{(j)}; C, \sigma_D)$ （深度读 RGB 观测 + 自身）
- $h_A^{(j+1)} = B_A^{(j)}(h_A^{(j)} \mid h_D^{(j)}; C, \sigma_D, \sigma_A)$ （动作读深度 + 自身）

## 核心要点

1. **单向性**: 信息只能从 RGB → Depth → Action 单向传播，反向不可达
2. **防止外观污染**: 动作流无法直接读取 RGB 特征，避免外观变化（颜色、纹理）干扰动作决策
3. **保留语义引导**: 深度流可以读取 RGB 观测特征，获得目标语义引导
4. **与 [[Mixture-of-Transformers]] 结合**: 通过多流共享视觉块 $B_V$ + 独立动作块 $B_A$ 实现

## 代表工作
- [[PrismWAM]]: 提出 Chain Attention 用于解耦 WAM 中语义规划与物理展开

## 相关概念
- [[Mixture-of-Transformers]]
- [[World Action Model]]
- [[Decoupled Representation]]
