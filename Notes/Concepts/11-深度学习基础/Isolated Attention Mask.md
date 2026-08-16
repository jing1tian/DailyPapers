---
type: concept
aliases: [隔离注意力掩码, Isolated Mask, Decoupled Attention Mask]
---

# Isolated Attention Mask

## 定义

在多专家 Transformer 联合训练场景中，通过精心设计的注意力掩码，让不同专家的 token 组之间**相互不可见**，但都可以注意共享的条件输入。从而实现训练时知识共享、推理时模块独立部署。

## 数学形式

设共享观测 token 集合为 $\mathbf{z}$，视频 token 集合为 $\mathbf{v}$，动作 token 集合为 $\mathbf{a}$，注意力可达关系定义为：

$$
\text{Attend}(\mathbf{v}) \to \mathbf{z}, \quad \text{Attend}(\mathbf{a}) \to \mathbf{z}, \quad \mathbf{v} \not\to \mathbf{a}, \quad \mathbf{a} \not\to \mathbf{v}
$$

与之对比：
- **双向掩码**: $\mathbf{v} \leftrightarrow \mathbf{a}$（双向可见，训练推理耦合）
- **单向掩码** (action→video): $\mathbf{a} \to \mathbf{v}$（动作可见视频，仍需部署视频分支）
- **隔离掩码**: $\mathbf{v} \not\leftrightarrow \mathbf{a}$（完全隔离，推理可丢弃任意分支）

## 核心要点

1. **训练期共享**: 两个专家都经过共享的条件输入（观测/文本等）的注意力，互相的知识通过共享表示传递，而非直接 token 通信。
2. **推理期解耦**: 由于动作 token 从未直接见过视频 token，推理时直接运行动作专家即可，视频专家完全可丢弃。
3. **不同于 Causal Masking**: 因果掩码是时序方向的单向可见；隔离掩码是模态方向的完全隔离。
4. **适用场景**: 任何希望"训练时多任务协同、推理时轻量部署"的多专家架构。

## 代表工作

- [[SimWAM]]: 最早将 Isolated Attention Mask 用于自动驾驶 WAM 的动作-视频解耦，实现 91.5 PDMS on NAVSIM。

## 相关概念

- [[MoT]] (Mixture of Transformers): 隔离掩码通常在 MoT 的多专家框架中实现
- [[Shared Attention]]: 共享注意力是知识传递的载体
- [[Cross-Modality Causal Masking]]: 跨模态因果掩码的相关变体
- [[FlowMatching]]: SimWAM 用 Flow Matching 作为训练目标
