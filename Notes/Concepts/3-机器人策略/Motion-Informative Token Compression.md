---
type: concept
aliases: [运动信息 Token 压缩, Motion-Relevance Token Compression]
---

# Motion-Informative Token Compression

## 定义
一种用于压缩世界模型生成的未来视角表示的技术：通过比较同一空间位置 token 嵌入在前后两帧的差异，筛选出"运动/变化"最显著的 Top-K 个 token，丢弃其余冗余 token，从而大幅降低生成视角的计算与延迟开销。

## 数学形式
对视角 $v$ 中位置 $j$ 的 token，用嵌入余弦距离计算运动相关性分数：

$$
\delta_{t,j}^v = 1 - \cos\left(E(c_{t,j}^v),\ E(c_{t+1,j}^v)\right), \quad j = 1, \ldots, N_{\text{tok}}
$$

取分数最高的 Top $K$ 个 token 保留，其余丢弃：

$$
\hat{o}_{t+1}^{\prime v} = \text{TopK}_{j}\left(\delta_{t,j}^v;\ K\right)
$$

## 核心要点
1. **运动相关性打分**: 用 [[余弦相似度]]衡量同一位置 token 嵌入在两帧间的变化幅度，而非简单的像素差异
2. **大幅压缩比**: 典型场景下从 625 个 token 压缩到 16 个（约 39×），多视角候选总量可从数千压缩到几十
3. **延迟换算直接**: token 数量减少直接带来生成延迟的线性下降（如 6–7 秒 → 0.2–0.3 秒/视角）
4. **保留语义关键信息**: 假设动作预测真正依赖的是场景中"发生变化"的局部区域，而非完整视角的所有像素

## 代表工作
- [[UniviewVLA]]: 提出该技术，将世界模型生成的未来辅助视角从 625 token 压缩到 16 token，使生成式多视角辅助信息可用于实时动作预测

## 相关概念
- [[VQ-VAE]]
- [[World Model]]
- [[余弦相似度]]
- [[Action-Entropy View Selection]]
