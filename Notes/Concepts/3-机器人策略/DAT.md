---
type: concept
aliases: [Discrete Action Tokenizer, 离散动作分词器]
---

# DAT（Discrete Action Tokenizer）

## 定义

DAT 是 [[PRIME W0]] 提出的离散动作分词器，通过注册瓶颈编码器 + [[FSQ|有限标量量化（FSQ）]] + 交叉注意力解码器，将连续机器人动作序列映射为固定长度离散 token，使动作表示与 VLM 的 token 范式对齐。

## 数学形式

多组件重建损失：

$$
\mathcal{L}_{DAT} = \text{SmoothL1}(\hat{p}, p) + \underbrace{\|\log(\hat{R}^T R)\|_F}_{d_{geo}} + \text{BCE}(\hat{g}, g) + \mathcal{L}_{freq} + \mathcal{L}_{dist}
$$

## 核心要点

1. **注册瓶颈编码器**: 非对称注意力结构，压缩动作序列为固定长度表示
2. **FSQ 量化**: 有限标量量化替代 VQ-VAE，产生稳定的固定长度离散 token（不同于 FAST 的可变长度）
3. **几何感知重建**: SO(3) 测地线距离监督旋转（$d_{geo} = \|\log(\hat{R}^T R)\|_F$），DCT 频域损失保证轨迹平滑
4. **双臂协调约束**: $\mathcal{L}_{dist}$ 监督左右臂末端执行器间距，鼓励协调性

## 代表工作

- [[PRIMEW0]]: DAT 的提出论文，Stage 2 NTP 预训练与 VLA 动作解码对齐

## 相关概念

- [[FSQ]]: DAT 使用的量化方法
- [[Action Chunking]]: DAT tokenize 的目标动作块
- [[FAST]]: 同类离散动作分词器（可变长度，DAT 改进为固定长度）
- [[Flow Matching]]: DAT 产生的 token 用于后续 ADM 流匹配解码器的训练
