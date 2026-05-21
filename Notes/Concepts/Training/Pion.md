---
type: concept
aliases: [Pion, sPectral hIgh-pass Optimization on momeNtum]
---

# Pion

## 定义
对 Muon 优化器的改进，通过保留梯度动量矩阵的谱高频分量（信息密集的主奇异值）、滤除低频噪声尾部，解决 Muon 在 VLA 微调和 RLVR 场景下的失效问题。

## 数学形式
$$\mathbf{M}_t^{\text{Pion}} = U \cdot \text{HighPassFilter}(\Sigma) \cdot V^\top$$

其中 $U\Sigma V^\top = \text{SVD}(\mathbf{M}_t)$，高通滤波器保留奇异值中前 $k$ 个主成分，截断低 SNR 的噪声尾部。

## 核心要点
1. **诊断 Muon 失效**：VLA action head 梯度低秩（erank 低）→ Muon 均匀谱白化放大噪声；RLVR GRPO 梯度 SNR 低 → 同样问题
2. **高通滤波策略**：保留 SVD 的高奇异值方向（信息），截断低奇异值噪声尾
3. 在 LIBERO（VLA）和 MATH/GRPO（RLVR）上均有改进
4. 与 [[AdamW]] 相比在 VLA 微调场景具有优势

## 代表工作
- Rethinking Muon Beyond Pretraining (2605.19282): 提出 Pion 优化器

## 相关概念
- [[GRPO]]
- [[VLA]]
- [[SFT]]
- [[Reinforcement Learning]]
