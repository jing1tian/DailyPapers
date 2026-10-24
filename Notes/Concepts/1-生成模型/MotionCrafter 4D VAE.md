---
type: concept
aliases: [MotionCrafter VAE, 4D VAE, 4D Flow VAE, MotionCrafter 4D 变分自编码器]
---

# MotionCrafter 4D VAE

## 定义

MotionCrafter 4D VAE 是一个专为 4D 时空流场（点图序列 + 帧间位移序列）设计的变分自编码器，将高维的 3D 场景流时序数据压缩为低维可去噪潜码，支持在扩散 Transformer 框架中对场景流进行生成建模。

## 核心要点

1. **双流编码**: 分别对点图序列 $P_{t:t+H}$（绝对 3D 坐标）和帧间位移序列 $F_{t:t+H}$（相对运动）独立编码，沿通道维度拼接
2. **冻结使用**: 在下游策略训练中编码器和解码器均冻结，仅作为特征提取器
3. **Patchify 策略**: 流 patchifier 每 token 合并两帧潜码，降低序列长度
4. **解码恢复**: 解码器 $\mathcal{D}$ 从潜码恢复 $P$ 和 $F$，用于可视化和监督验证

## 数学形式

$$
m = [\mathcal{E}_P(P_{t:t+H});\ \mathcal{E}_F(F_{t:t+H})]
$$

$$
(\hat{P}, \hat{F}) = \mathcal{D}(m)
$$

## 代表工作

- [[UniFlowWAM]]: 使用冻结 MotionCrafter 4D VAE 将 Track4World 提取的伪标签场景流压缩为流专家可处理的潜码

## 相关概念

- [[场景流]]: 被编码的数据类型
- [[Causal VAE]]: 同类时序 VAE 架构
- [[World Action Model]]: 使用场景
