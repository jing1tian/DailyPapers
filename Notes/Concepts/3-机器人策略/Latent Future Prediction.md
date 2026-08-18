---
type: concept
aliases: [潜空间未来预测, 未来潜在预测, 潜在未来对齐]
---

# Latent Future Prediction

## 定义

在语义潜空间中预测未来视觉帧表示的自监督辅助任务：模型在当前步预测后续若干帧的视觉特征，以此促使模型学习动态场景的时序演化规律，增强前瞻性决策能力。

## 数学形式

预测：
$$
\hat{\mathbf{y}}_{t+i} = f_{\text{pred}}(\mathbf{h}^{\text{future}}_{i}), \quad i = 1, \ldots, H
$$

目标（冻结编码器）：
$$
\mathbf{y}_{t+i} = \phi_{\text{frozen}}(\mathbf{o}_{t+i}), \quad \mathbf{y}_{t+i} \in \mathbb{R}^{d}
$$

损失（掩码余弦相似度）：
$$
\mathcal{L}_{\text{future}} = \frac{\sum_{i=1}^{H} m_i \left[1 - \cos\left(\hat{\mathbf{y}}_{t+i}, \mathbf{y}_{t+i}\right)\right]}{\sum_{i=1}^{H} m_i}
$$

## 核心要点

1. **冻结目标编码器**：使用冻结的大型视觉编码器（如 DINOv3）作为预测目标，若目标可训练则训练崩溃（-31.9%）
2. **语义空间预测**：在高维语义特征空间预测而非像素空间，计算开销低
3. **联合训练**：$\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}}$，辅助任务不影响主任务推理速度
4. **前瞻性增强**：迫使模型内部表示编码物体运动轨迹，而不仅仅是当前状态

## 代表工作

- [[ReflexVLA]]：冻结 DINOv3 目标，+26% 成功率提升
- [[Future Latent Alignment]]：类似思路，面向世界模型场景

## 相关概念

- [[DINOv3]]
- [[Stop-Gradient]]
- [[Self-Supervised Learning]]
- [[Predictive-Coding]]
- [[Learnable Foresight Token]]
