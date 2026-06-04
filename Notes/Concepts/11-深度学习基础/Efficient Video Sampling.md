---
type: concept
aliases: [EVS, 高效视频采样, 时序 token 剪枝]
---

# Efficient Video Sampling (EVS)

## 定义

Efficient Video Sampling（EVS）是一种即插即用的视频 token 冗余剪枝方法，通过识别并移除跨帧在空间上保持静止的 patch（时序静态区域），减少送入 Transformer 的 token 总量，从而降低计算开销，无需修改模型架构或重新训练。

## 核心要点

1. **时序冗余识别**: 比较相邻帧中每个空间 patch 的变化量，将变化量低于阈值的 patch 标记为"时序静态"
2. **选择性剪枝**: 对时序静态 patch 在时间轴上只保留关键帧的 token，其余帧的对应 token 直接跳过
3. **位置标识保留**: 剪枝后仍保留 token 的绝对位置编码（通过 mRoPE），保证模型对未剪枝 token 的时序感知正确
4. **即插即用**: 不依赖特定架构，不需要重新训练，可作为推理加速插件叠加在任何支持位置编码的视频 Transformer 上
5. **与 EVS 前代工作的关系**: 对应于 token merging / token dropping 的视频特化版本，核心创新在于利用物理时序静态性而非学习相似度

## 数学形式

设第 $t$ 帧 patch $p$ 的特征为 $f_t^p$，判断静态条件：

$$
\|f_t^p - f_{t-1}^p\|_2 < \delta \implies \text{prune token } (t, p)
$$

其中 $\delta$ 为静态判断阈值，剪枝后的 token 集合送入 Transformer，其余位置通过插值或跳过恢复。

## 代表工作

- [[Cosmos3]]: 在 Cosmos 3 中引入 EVS 作为视频推理的 token 压缩方案，在不重新训练的前提下加速长视频推理

## 相关概念

- [[Visual Tokenizer]]
- [[Rotary Position Embedding]]
- [[Diffusion Transformer]]
- [[Mixture-of-Transformers]]
