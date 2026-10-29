---
type: concept
aliases: [先见损失, Foresight Supervision, Future Representation Loss]
---

# Foresight Loss

## 定义

一种辅助训练目标：通过可学习的预测头将当前观测特征映射到冻结未来锚编码的表示，以余弦距离作为损失，训练结束后丢弃预测头，无需在训练或推理时生成完整未来帧。

## 数学形式

$$
\mathcal{L}_{\mathrm{F}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \mathbb{E}_{n} \Big[ 1 - \cos\big(\hat{\mathbf{g}}_{k,n},\ \operatorname{sg}[\mathbf{g}_{k,n}]\big) \Big]
$$

其中：
- $\mathcal{K}$：目标帧偏移集合（如 $\{1, 2\}$ 环境步）
- $\hat{\mathbf{g}}_{k,n}$：预测头对偏移 $k$、位置 $n$ 的输出
- $\mathbf{g}_{k,n}$：冻结未来锚对第 $k$ 步未来帧的编码特征
- $\operatorname{sg}[\cdot]$：[[Stop-Gradient|stop-gradient]] 算子

## 核心要点

1. **无生成**：不需要解码或生成完整未来帧，仅预测特征空间中的表示
2. **推理时丢弃**：预测头和未来锚仅在训练时存在，推理时策略与纯动作策略完全相同
3. **锚选择灵活**：未来锚可以是冻结骨干副本、VAE 编码器或随机初始化网络；骨干预训练比锚预训练更重要
4. **偏移量敏感**：未来帧偏移量的选择对性能有显著影响，当前帧偏移（$k=0$）对部分骨干有害

## 代表工作

- [[JAM]]: 提出先见损失替代 WAM 中的视觉生成目标，在 LIBERO/LIBERO-Plus/RoboTwin 上取得 SOTA

## 相关概念

- [[World Action Model]]: 先见损失所改进的训练范式
- [[Stop-Gradient]]: 先见损失中阻止梯度流向未来锚的算子
- [[Flow Matching]]: JAM 中与先见损失共同使用的动作生成方法
- [[Cosine Similarity]]: 先见损失使用余弦距离衡量预测与目标的相似性
