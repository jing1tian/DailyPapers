---
type: concept
aliases: [RoPE, 旋转位置编码, Rotary Position Embedding]
---

# Rotary Position Encoding

## 定义

一种通过旋转矩阵将位置信息编码到 Query/Key 向量中的位置编码方法，能够自然地表达相对位置关系，在 Transformer 中广泛使用。

## 数学形式

标准 RoPE 对位置 $m$ 的编码：

$$
f(\mathbf{x}, m) = \mathbf{x} e^{im\theta}
$$

HY-World 2.0 中的**归一化 RoPE**（用于跨分辨率一致性）：

$$
\hat{x}_i = \frac{2i + 1}{H_p} - 1, \quad \hat{y}_j = \frac{2j + 1}{W_p} - 1
$$

将 patch 坐标归一化到 $[-1, 1]$，将分辨率外推转化为插值。

## 核心要点

1. 通过旋转操作编码绝对位置，注意力计算时自然呈现相对位置
2. 外推到训练时未见分辨率时，标准 RoPE 性能退化明显
3. 归一化 RoPE 将位置归一化到固定范围，实现跨分辨率高度一致（余弦相似度 >0.95）
4. 在视觉 Transformer 中，二维 RoPE 分别处理水平和垂直方向

## 代表工作

- [[HYWorld2]]: WorldMirror 2.0 使用归一化 RoPE 支持任意分辨率推理
- RoFormer: 原始 RoPE 论文

## 相关概念

- [[Video Diffusion Transformer]]
- [[Multi-Modal Diffusion Transformer]]
