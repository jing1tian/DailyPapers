---
type: concept
aliases: [Future Chunk Prediction, 未来块预测]
---

# FCP

## 定义
Future Chunk Prediction（FCP）是一种用于自回归视频扩散模型的辅助训练目标，通过强迫模型预测时间上分离的未来帧块来克服 teacher forcing 下的时间冗余捷径问题。

## 数学形式
在标准 teacher-forced 去噪目标 $\mathcal{L}_\text{TF}$ 之外，额外引入未来块预测损失：
$$\mathcal{L} = \mathcal{L}_\text{TF} + \lambda \cdot \mathcal{L}_\text{FCP}$$

其中 $\mathcal{L}_\text{FCP}$ 要求模型预测与当前 chunk 时间上分离（间隔 $\Delta t$ 步）的未来 chunk。

## 核心要点
1. 解决 Teacher Forcing 捷径问题：AR 视频模型在高时间冗余下不需要建模真正的场景演化
2. 强迫模型学习跨 chunk 的长程演化规律，而非依赖局部时间连续性
3. 即插即用的辅助目标，与标准 AR 视频扩散训练兼容
4. 对 WAM 等世界模型的长程一致性有潜在迁移价值

## 代表工作
- [[FCP]]/Looking Ahead: 提出 FCP 的原始论文，在 AR 视频扩散模型上验证

## 相关概念
- [[Teacher Forcing]]
- [[Diffusion Model]]
- [[World Model]]
