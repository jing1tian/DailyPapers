---
type: concept
aliases: [Centered Kernel Alignment, 中心化核对齐]
---

# CKA (Centered Kernel Alignment)

## 定义
用于比较两个神经网络表示（特征矩阵）之间相似性的度量，对正交变换和各向同性缩放不变。

## 数学形式
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K,K) \cdot \text{HSIC}(L,L)}}$$

其中 $K, L$ 是 Gram 矩阵（相似度矩阵），HSIC 是 Hilbert-Schmidt Independence Criterion。

## 核心要点
1. 输出值在 [0, 1]，1 表示两个表示完全线性等价
2. 可用于比较同一网络不同层、不同网络同一位置、训练前后的表示变化
3. **Checkpoint drift**：用 CKA 追踪训练过程中表示的漂移情况，可诊断 VLA 训练稳定性
4. 线性 CKA（使用线性核）比 RBF 核计算更快，在实践中最常用

## 代表工作
- [[VLA-Trace]]: 用 Cross-Modal CKA 分析 VLA 内部表示动力学
- [[ElegantVLA]]: 用 CKA 分析何时 thinking 对 VLA 有必要

## 相关概念
- [[Transformer]] — 通常用于分析各层 attention 表示
- [[VLA（视觉-语言-动作模型）]]
