---
type: concept
aliases: [分层分块, 层次化时序压缩]
---

# Hierarchical Chunking

## 定义
一种将序列数据通过多层渐进式压缩提升时序抽象粒度的方法，每层通过可学习的边界检测对序列进行自适应分块，逐步从帧级运动聚合为技能级表示。

## 数学形式

$$
Z^{(s+1)} = \text{Chunk}_s(E_s(Z^{(s)}); b^{(s)}), \quad s = 1, \ldots, H
$$

其中 $E_s$ 为第 $s$ 层编码器，$b^{(s)}$ 为该层的边界标签，$H$ 为总层次数。

## 核心要点
1. 边界通过相邻 token 的余弦不相似度评分（dissimilarity scoring）检测
2. 目标边界比率 $\rho_s$ 控制每层的压缩程度
3. 多层嵌套：低层捕获短时运动转换，高层捕获任务阶段切换
4. 与记忆写入门对齐：高层边界触发记忆更新

## 代表工作
- [[HiMem-WAM]]: Stage II 中用于从低层潜变量序列发现高层技能表示

## 相关概念
- [[Hierarchical Latent Action]]
- [[Attention Pooling]]
- [[Memory Gating]]
