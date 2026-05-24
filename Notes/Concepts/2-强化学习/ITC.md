---
type: concept
aliases: [Identifiable Token Correspondence, ITC decoding, OT-based world model decoding]
---

# ITC

## 定义

ITC（Identifiable Token Correspondence）是一种即插即用的世界模型解码机制，通过最优传输将下一帧 token 预测建模为结构化分配问题，让每个 token 选择性地从前一帧复制或重新生成，从根本上消除长程滚动中的对象重影和消失问题。

## 数学形式

核心解码规则：

$$
u'_j = \begin{cases} u_i & \text{if } \Pi^{(\text{prev})}_{ij} = 1 \\ \text{sample}(p_j) & \text{if } \Pi^{(\text{gen})}_{jj} = 1 \end{cases}
$$

前帧亲和矩阵：$A^{(\text{prev})}_{ij} = \langle p_j, u_i \rangle - c_d \cdot D\bigl((x_i, y_i), (x_j, y_j)\bigr)$

## 核心要点

1. **即插即用**: 不修改 Transformer 架构或训练流程，插入 OT 求解层即可
2. **OT 框架**: 用 Sinkhorn 算法求软分配，再二值化为硬 0-1 分配
3. **低开销**: 额外计算开销仅 2.8%（单卡 RTX 3090）
4. **跨 Tokenizer 泛化**: 在 Patch-lookup 和 VQ-VAE 两类 tokenizer 上均有效

## 代表工作

- [[ITC]]（本方法论文）: Craftax-classic 72.5% Return，Atari 100K IQM 1.092（2025 SOTA）

## 相关概念

- [[最优传输]]
- [[Sinkhorn 算法]]
- [[二值化（Binarization）]]
- [[旋转位置编码]]
- [[World Model]]
- [[IRIS]]
