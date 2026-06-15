---
type: concept
aliases: [Frequency-space Action Sequence Tokenizer, 频域动作tokenizer]
---

# FAST

## 定义
Frequency-space Action Sequence Tokenizer，将机器人动作序列转换到频域后进行 tokenization，通过 DCT（离散余弦变换）压缩动作 chunk，显著减少 token 数量同时保留动作的主要模式。

## 数学形式
对动作序列 $a_{1:T} \in \mathbb{R}^{T \times d}$ 进行 DCT 变换后量化：
$$\tilde{a} = \text{DCT}(a_{1:T}), \quad \text{token} = Q(\tilde{a}[:K])$$
取低频分量 $K \ll T$ 保留主要动作模式。

## 核心要点
1. **频域压缩**：低频分量捕获动作的主要轨迹，高频分量为细节噪声
2. **Token 效率**：比逐帧 tokenization 减少 10× 以上 token 数量
3. **与 [[ACT]] 对比**：ACT 用 CVAE 压缩，FAST 用频域压缩，更适合 autoregressive VLA

## 代表工作
- [[π₀.₅]] / Pi0-FAST: FAST 的主要使用场景
- [[LabVLA]]: 使用 FAST tokenizer 处理科学实验室操作序列
- [[AIR-VLA+]]: 在空中操作系统中使用 FAST

## 相关概念
- [[ACT]]
- [[Action Chunking]]
- [[OpenVLA]]
