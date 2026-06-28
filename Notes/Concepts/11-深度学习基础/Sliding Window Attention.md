---
type: concept
aliases: [SWA, Dilated Sliding Window Attention, DSWA, 滑窗注意力, 扩张滑窗注意力]
---

# Sliding Window Attention

## 定义
一种将标准全局自注意力的注意力范围限制在固定大小窗口内的局部注意力机制：每个 token 只与其前后 $w/2$ 范围内的 token 计算注意力，把计算和显存复杂度从序列长度的平方 $O(n^2)$ 降为线性 $O(n \cdot w)$；其扩张变体 Dilated Sliding Window Attention (DSWA) 保持窗口大小不变，但引入扩张因子 $d$ 使窗口内的位置间隔变大，在不增加计算量的前提下扩展有效感受野以覆盖更长的中程依赖。

## 数学形式
$$
\mathrm{SWA}(Q,K,V)_i = \sum_{j \in [i-\frac{w}{2}, i+\frac{w}{2}]} \mathrm{Softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right) V_j
$$

其中 $w$ 是窗口大小；DSWA 在此基础上将求和范围替换为按扩张因子 $d$ 采样的位置子集 $\{i + d\cdot k \mid k \in [-w/2, w/2]\}$，用相同的 $w$ 个 key/value 覆盖约 $d \cdot w$ 倍的时序跨度。

## 核心要点
1. 把注意力计算限制在局部窗口内，是长序列/长视频建模中替代全局注意力以获得线性复杂度的最常用手段之一
2. 单纯堆叠固定窗口的 SWA 只能捕获短程依赖，需要配合扩张（DSWA，逐层扩大有效跨度）或全局记忆模块（如 GLA）才能覆盖长程信息，否则会因为局部窗口不可测（non-measurable）而产生不可消除的超额风险
3. 通常与 [[RoPE]] 等相对位置编码配合，编码窗口内 token 间的相对时序/空间关系
4. 在 Hybrid Linear Temporal Attention 类架构中承担"局部管运动细节、全局管长程一致性"的职责分工，与 [[GLA]] 互补

## 代表工作
- [[Kairos]]: 用 SWA（$d=1$）建模短程局部运动、DSWA（$d \in \{6,12\}$）建模约 1 秒级别中程依赖，与 GLA 全局记忆组合成 Hybrid Linear Temporal Attention，并从理论上证明纯局部窗口注意力存在不可消除的超额风险下界（必须引入持久化全局状态）

## 相关概念
- [[GLA]]
- [[RoPE]]
- [[Mixture-of-Transformers]]
